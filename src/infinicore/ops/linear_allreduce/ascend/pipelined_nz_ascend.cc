#include "infinicore/context/context.hpp"
#include <acl/acl.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <infiniccl.h>
#include <infiniop.h>
#include <memory>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace infinicore::op::linear_allreduce_impl::ascend {
namespace {
void check(int code, const char *operation) {
    if (code) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
    }
}
struct DeviceScope {
    int32_t previous = -1;
    explicit DeviceScope(int32_t device) {
        aclrtGetDevice(&previous);
        aclrtSetDevice(device);
    }
    ~DeviceScope() {
        if (previous >= 0) {
            aclrtSetDevice(previous);
        }
    }
};
struct CommunicationStream {
    aclrtStream stream = nullptr;
    int32_t device = -1;
    CommunicationStream() {
        check(aclrtGetDevice(&device), "get pipeline device");
        check(aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_FAST_LAUNCH), "create pipeline stream");
    }
    ~CommunicationStream() {
        DeviceScope scope(device);
        if (stream) {
            aclrtSynchronizeStream(stream);
            aclrtDestroyStream(stream);
        }
    }
};
std::shared_ptr<CommunicationStream> communication_stream() {
    static thread_local std::unordered_map<void *, std::weak_ptr<CommunicationStream>> streams;
    auto &entry = streams[context::getStream()];
    auto stream = entry.lock();
    if (!stream) {
        stream = std::make_shared<CommunicationStream>();
        entry = stream;
    }
    return stream;
}
struct Chunk {
    graph::GraphTensor input, output;
    std::optional<graph::GraphTensor> workspace;
    infiniopGemmDescriptor_t descriptor = nullptr;
    aclrtEvent ready = nullptr;
    int32_t device = -1;
    Chunk(const Tensor &a, const Tensor &b, const Tensor &c)
        : input(a), output(c) {
        check(aclrtGetDevice(&device), "get chunk device");
        check(infiniopCreateGemmNzDescriptor(context::getInfiniopHandle(a->device()), &descriptor, c->desc(), a->desc(), b->desc()), "plan pipeline GEMM");
        try {
            size_t bytes = 0;
            check(infiniopGetGemmWorkspaceSize(descriptor, &bytes), "size pipeline GEMM");
            workspace.emplace(Tensor::empty({std::max<size_t>(1, bytes)}, DataType::U8, a->device()));
            // Ex events snapshot each record independently and also participate
            // in cross-stream native graph capture. No manual reset is needed.
            check(aclrtCreateEventExWithFlag(&ready, ACL_EVENT_SYNC), "create pipeline ready event");
        } catch (...) {
            infiniopDestroyGemmDescriptor(descriptor);
            descriptor = nullptr;
            throw;
        }
    }
    ~Chunk() {
        DeviceScope scope(device);
        if (descriptor) {
            infiniopDestroyGemmDescriptor(descriptor);
        }
        if (ready) {
            aclrtDestroyEvent(ready);
        }
    }
};
class PipelinedNz final : public graph::GraphOperator {
    graph::GraphTensor weight_;
    infinicclComm_t communicator_;
    std::shared_ptr<CommunicationStream> communication_;
    std::vector<std::unique_ptr<Chunk>> chunks_;
    aclrtEvent finish_ = nullptr;

public:
    PipelinedNz(Tensor output, const Tensor &input, const Tensor &weight, infinicclComm_t communicator)
        : weight_(weight), communicator_(communicator), communication_(communication_stream()) {
        constexpr size_t chunk_size = 1024;
        for (size_t start = 0; start < input->size(0); start += chunk_size) {
            const size_t rows = std::min(chunk_size, input->size(0) - start);
            chunks_.push_back(std::make_unique<Chunk>(input->narrow({{0, start, rows}}), weight, output->narrow({{0, start, rows}})));
        }
        check(aclrtCreateEventExWithFlag(&finish_, ACL_EVENT_SYNC), "create pipeline finish event");
    }
    ~PipelinedNz() override {
        DeviceScope scope(communication_->device);
        // All chunks share one communication stream per rank/main stream.
        // Retire its outstanding reads before releasing graph-owned buffers.
        aclrtSynchronizeStream(communication_->stream);
        if (finish_) {
            aclrtDestroyEvent(finish_);
        }
    }
    void run() const override {
        auto main = static_cast<aclrtStream>(context::getStream());
        auto auxiliary = communication_->stream;
        for (const auto &chunk : chunks_) {
            check(infiniopGemm(chunk->descriptor, chunk->workspace.value()->data(), chunk->workspace.value()->numel(), chunk->output->data(), chunk->input->data(), weight_->data(), 1.f, 0.f, main), "pipeline GEMM");
            check(aclrtRecordEvent(chunk->ready, main), "record pipeline ready");
            check(aclrtStreamWaitEvent(auxiliary, chunk->ready), "wait pipeline GEMM");
            check(infinicclAllReduce(chunk->output->data(), chunk->output->data(), chunk->output->numel(), INFINI_DTYPE_BF16, INFINICCL_SUM, communicator_, auxiliary), "pipeline allreduce");
        }
        check(aclrtRecordEvent(finish_, auxiliary), "record pipeline finish");
        check(aclrtStreamWaitEvent(main, finish_), "join pipeline allreduce");
    }
};
} // namespace
bool try_record_pipelined_nz(Tensor output, const Tensor &input, const Tensor &weight, infinicclComm_t communicator) {
    const char *flag = std::getenv("INFINICORE_ASCEND_PIPELINED_NZ_REDUCE");
    if (!flag || std::strcmp(flag, "1") != 0 || !context::isGraphRecording()
        || input->device().getType() != Device::Type::ASCEND
        || input->dtype() != DataType::BF16 || weight->dtype() != DataType::BF16
        || output->dtype() != DataType::BF16 || !communicator
        || input->ndim() != 2 || weight->ndim() != 2 || output->ndim() != 2
        || !input->is_contiguous() || !weight->is_contiguous() || !output->is_contiguous()
        || input->size(0) <= 1024 || input->size(0) > 8192
        || input->size(1) != weight->size(0) || output->size(0) != input->size(0)
        || output->size(1) != weight->size(1)
        || weight->size(0) % 16 || weight->size(1) % 16) {
        return false;
    }
    context::addGraphOperator(std::make_shared<PipelinedNz>(output, input, weight, communicator));
    static thread_local bool reported = false;
    if (!reported) {
        spdlog::info("Ascend NZ prefill pipeline recorded: rows={}, chunk_rows=1024", input->size(0));
        reported = true;
    }
    return true;
}
} // namespace infinicore::op::linear_allreduce_impl::ascend
