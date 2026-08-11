#ifdef ENABLE_ASCEND_API

#include "infinicore/ops/matmul_allreduce_ascend.hpp"
#include "../../../infiniccl/infiniccl_impl.h"
#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/level2/aclnn_matmul_all_reduce.h>

extern "C" int HcclGetCommName(
    void *communicator, char *communicator_name);

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MatmulAllReduceAscend);

namespace matmul_allreduce_ascend_impl {

static aclDataType to_acl_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::F16:
        return ACL_FLOAT16;
    case DataType::BF16:
        return ACL_BF16;
    default:
        throw std::runtime_error(
            "[matmul_allreduce/ascend] only fp16 and bf16 are supported");
    }
}

static aclTensor *make_acl_tensor_2d(const Tensor &tensor) {
    const int64_t dims[2] = {
        static_cast<int64_t>(tensor->shape()[0]),
        static_cast<int64_t>(tensor->shape()[1])};
    const int64_t strides[2] = {
        static_cast<int64_t>(tensor->strides()[0]),
        static_cast<int64_t>(tensor->strides()[1])};
    return aclCreateTensor(
        dims,
        2,
        to_acl_dtype(tensor->dtype()),
        strides,
        0,
        ACL_FORMAT_ND,
        dims,
        2,
        const_cast<void *>(
            reinterpret_cast<const void *>(tensor->data())));
}

struct StreamWorkspace {
    void *ptr = nullptr;
    uint64_t capacity = 0;
    std::vector<void *> retired;
};

static void *acquire_stream_workspace(
    aclrtStream stream, uint64_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    // Rank workers own their stream. Avoid a process-wide mutex on every
    // fused row-parallel layer in the decode path.
    thread_local std::unordered_map<aclrtStream, StreamWorkspace> workspaces;
    auto &workspace = workspaces[stream];
    if (workspace.capacity < bytes) {
        void *new_ptr = nullptr;
        auto ret = aclrtMalloc(&new_ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(
                "[matmul_allreduce/ascend] workspace allocation failed: "
                + std::to_string(ret));
        }
        if (workspace.ptr != nullptr) {
            // Calls on one rank stream are ordered. Keep old workspaces alive
            // because freeing here would introduce a host synchronization.
            workspace.retired.push_back(workspace.ptr);
        }
        workspace.ptr = new_ptr;
        workspace.capacity = bytes;
    }
    return workspace.ptr;
}

static const std::string &get_group_name(
    infinicclComm_t communicator) {
    if (communicator == nullptr || communicator->comm == nullptr) {
        throw std::runtime_error(
            "[matmul_allreduce/ascend] communicator is null");
    }
    // The HCCL communicator name is immutable for the model lifetime. vLLM
    // also resolves it once, rather than once per layer invocation.
    thread_local std::unordered_map<infinicclComm_t, std::string> names;
    auto found = names.find(communicator);
    if (found != names.end()) {
        return found->second;
    }
    char name[128] = {};
    auto ret = HcclGetCommName(
        communicator->comm, name);
    if (ret != 0) {
        throw std::runtime_error(
            "[matmul_allreduce/ascend] HcclGetCommName failed: "
            + std::to_string(ret));
    }
    return names.emplace(communicator, name).first->second;
}

struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor input;
    graph::GraphTensor weight;
    infinicclComm_t communicator;
};

void *plan(
    Tensor out,
    const Tensor &input,
    const Tensor &weight_transposed,
    infinicclComm_t communicator) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(weight_transposed),
        communicator};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    infinicore::context::setDevice(p->input->device());

    if (p->input->ndim() != 2 || p->weight->ndim() != 2
        || p->out->ndim() != 2) {
        throw std::runtime_error(
            "[matmul_allreduce/ascend] expected 2D input, weight and output");
    }
    if (p->input->shape()[1] != p->weight->shape()[0]
        || p->out->shape()[0] != p->input->shape()[0]
        || p->out->shape()[1] != p->weight->shape()[1]) {
        throw std::runtime_error(
            "[matmul_allreduce/ascend] incompatible matrix shapes");
    }
    if (!p->input->is_contiguous() || !p->out->is_contiguous()) {
        throw std::runtime_error(
            "[matmul_allreduce/ascend] input and output must be contiguous");
    }
    if (p->input->dtype() != p->weight->dtype()
        || p->input->dtype() != p->out->dtype()) {
        throw std::runtime_error(
            "[matmul_allreduce/ascend] tensor dtypes must match");
    }

    Tensor input(p->input);
    Tensor weight(p->weight);
    Tensor out(p->out);
    auto *input_acl = make_acl_tensor_2d(input);
    auto *weight_acl = make_acl_tensor_2d(weight);
    auto *out_acl = make_acl_tensor_2d(out);
    if (input_acl == nullptr || weight_acl == nullptr || out_acl == nullptr) {
        if (input_acl != nullptr) {
            aclDestroyTensor(input_acl);
        }
        if (weight_acl != nullptr) {
            aclDestroyTensor(weight_acl);
        }
        if (out_acl != nullptr) {
            aclDestroyTensor(out_acl);
        }
        throw std::runtime_error(
            "[matmul_allreduce/ascend] aclCreateTensor failed");
    }

    const std::string &group = get_group_name(p->communicator);
    static const int64_t comm_turn = []() {
        const char *value = std::getenv("INFINICORE_ASCEND_MATMUL_ALLREDUCE_COMM_TURN");
        return value == nullptr ? int64_t{0}
                                : std::strtoll(value, nullptr, 10);
    }();
    static const int64_t stream_mode = []() {
        const char *value = std::getenv("INFINICORE_ASCEND_MATMUL_ALLREDUCE_STREAM_MODE");
        return value == nullptr ? int64_t{1}
                                : std::strtoll(value, nullptr, 10);
    }();

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnMatmulAllReduceGetWorkspaceSize(
        input_acl,
        weight_acl,
        nullptr,
        group.c_str(),
        "sum",
        comm_turn,
        stream_mode,
        out_acl,
        &workspace_size,
        &executor);
    if (ret != ACL_SUCCESS) {
        const char *message = aclGetRecentErrMsg();
        aclDestroyTensor(input_acl);
        aclDestroyTensor(weight_acl);
        aclDestroyTensor(out_acl);
        throw std::runtime_error(
            "[matmul_allreduce/ascend] "
            "aclnnMatmulAllReduceGetWorkspaceSize failed: "
            + std::to_string(ret) + ", "
            + (message != nullptr ? message : "(no ACL error)"));
    }

    auto stream = reinterpret_cast<aclrtStream>(infinicore::context::getStream());
    void *workspace = acquire_stream_workspace(stream, workspace_size);
    ret = aclnnMatmulAllReduce(
        workspace, workspace_size, executor, stream);

    aclDestroyTensor(input_acl);
    aclDestroyTensor(weight_acl);
    aclDestroyTensor(out_acl);

    if (ret != ACL_SUCCESS) {
        const char *message = aclGetRecentErrMsg();
        throw std::runtime_error(
            "[matmul_allreduce/ascend] aclnnMatmulAllReduce failed: "
            + std::to_string(ret) + ", "
            + (message != nullptr ? message : "(no ACL error)"));
    }
}

void cleanup(void **planned_meta_ptr) {
    auto *p = *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    delete p;
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MatmulAllReduceAscend::plan_dispatcher().registerDevice(
        Device::Type::ASCEND, &plan);
    MatmulAllReduceAscend::run_dispatcher().registerDevice(
        Device::Type::ASCEND, &run);
    MatmulAllReduceAscend::cleanup_dispatcher().registerDevice(
        Device::Type::ASCEND, &cleanup);
    return true;
}();

} // namespace matmul_allreduce_ascend_impl

MatmulAllReduceAscend::MatmulAllReduceAscend(
    Tensor out,
    const Tensor &input,
    const Tensor &weight_transposed,
    infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        out, input, weight_transposed);
    INFINICORE_GRAPH_OP_DISPATCH(
        out->device().getType(),
        out,
        input,
        weight_transposed,
        communicator);
}

void MatmulAllReduceAscend::execute(
    Tensor out,
    const Tensor &input,
    const Tensor &weight_transposed,
    infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MatmulAllReduceAscend,
        out,
        input,
        weight_transposed,
        communicator);
}

Tensor matmul_allreduce_ascend(
    const Tensor &input,
    const Tensor &weight_transposed,
    infinicclComm_t communicator) {
    if (input->ndim() != 2 || weight_transposed->ndim() != 2) {
        throw std::runtime_error(
            "[matmul_allreduce/ascend] expected 2D matrices");
    }
    auto out = Tensor::empty(
        {input->shape()[0], weight_transposed->shape()[1]},
        input->dtype(),
        input->device());
    MatmulAllReduceAscend::execute(
        out, input, weight_transposed, communicator);
    return out;
}

} // namespace infinicore::op

#endif
