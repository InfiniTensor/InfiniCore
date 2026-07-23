#include "infinicore/ops/distributed/p2p.hpp"
#include "../../utils.hpp"

namespace infinicore::op::distributed {
namespace {
struct PlannedMeta {
    graph::GraphTensor tensor;
    int peer;
    infinicclComm_t communicator;
};
struct GroupedPlannedMeta {
    std::vector<graph::GraphTensor> tensors;
    int peer;
    infinicclComm_t communicator;
};
} // namespace

Send::Send(const Tensor &input, int peer, infinicclComm_t communicator) {
    INFINICORE_ASSERT(input->is_contiguous());
    planned_meta_ = new PlannedMeta{
        graph::GraphTensor(input), peer, communicator};
}

Send::~Send() {
    delete reinterpret_cast<PlannedMeta *>(planned_meta_);
}

void Send::run() const {
    const auto *meta = reinterpret_cast<PlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclSend(
        meta->tensor->data(), meta->tensor->numel(),
        static_cast<infiniDtype_t>(static_cast<int>(meta->tensor->dtype())),
        meta->peer, meta->communicator, infinicore::context::getStream()));
}

void Send::execute(
    const Tensor &input, int peer, infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Send, input, peer, communicator);
}

Recv::Recv(Tensor output, int peer, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->is_contiguous());
    planned_meta_ = new PlannedMeta{
        graph::GraphTensor(output), peer, communicator};
}

Recv::~Recv() {
    delete reinterpret_cast<PlannedMeta *>(planned_meta_);
}

void Recv::run() const {
    const auto *meta = reinterpret_cast<PlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclRecv(
        const_cast<void *>(static_cast<const void *>(meta->tensor->data())),
        meta->tensor->numel(),
        static_cast<infiniDtype_t>(static_cast<int>(meta->tensor->dtype())),
        meta->peer, meta->communicator, infinicore::context::getStream()));
}

void Recv::execute(
    Tensor output, int peer, infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Recv, output, peer, communicator);
}

GroupedSend::GroupedSend(const std::vector<Tensor> &inputs, int peer,
                         infinicclComm_t communicator) {
    std::vector<graph::GraphTensor> tensors;
    tensors.reserve(inputs.size());
    for (const auto &input : inputs) {
        INFINICORE_ASSERT(input->is_contiguous());
        tensors.emplace_back(input);
    }
    planned_meta_ = new GroupedPlannedMeta{
        std::move(tensors), peer, communicator};
}

GroupedSend::~GroupedSend() {
    delete reinterpret_cast<GroupedPlannedMeta *>(planned_meta_);
}

void GroupedSend::run() const {
    const auto *meta = reinterpret_cast<GroupedPlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclGroupStart(meta->communicator));
    try {
        for (const auto &tensor : meta->tensors) {
            INFINICORE_CHECK_ERROR(infinicclSend(
                tensor->data(), tensor->numel(),
                static_cast<infiniDtype_t>(static_cast<int>(tensor->dtype())),
                meta->peer, meta->communicator,
                infinicore::context::getStream()));
        }
    } catch (...) {
        (void)infinicclGroupEnd(meta->communicator);
        throw;
    }
    INFINICORE_CHECK_ERROR(infinicclGroupEnd(meta->communicator));
}

void GroupedSend::execute(const std::vector<Tensor> &inputs, int peer,
                          infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        GroupedSend, inputs, peer, communicator);
}

GroupedRecv::GroupedRecv(const std::vector<Tensor> &outputs, int peer,
                         infinicclComm_t communicator) {
    std::vector<graph::GraphTensor> tensors;
    tensors.reserve(outputs.size());
    for (const auto &output : outputs) {
        INFINICORE_ASSERT(output->is_contiguous());
        tensors.emplace_back(output);
    }
    planned_meta_ = new GroupedPlannedMeta{
        std::move(tensors), peer, communicator};
}

GroupedRecv::~GroupedRecv() {
    delete reinterpret_cast<GroupedPlannedMeta *>(planned_meta_);
}

void GroupedRecv::run() const {
    const auto *meta = reinterpret_cast<GroupedPlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclGroupStart(meta->communicator));
    try {
        for (const auto &tensor : meta->tensors) {
            INFINICORE_CHECK_ERROR(infinicclRecv(
                const_cast<void *>(static_cast<const void *>(tensor->data())),
                tensor->numel(),
                static_cast<infiniDtype_t>(static_cast<int>(tensor->dtype())),
                meta->peer, meta->communicator,
                infinicore::context::getStream()));
        }
    } catch (...) {
        (void)infinicclGroupEnd(meta->communicator);
        throw;
    }
    INFINICORE_CHECK_ERROR(infinicclGroupEnd(meta->communicator));
}

void GroupedRecv::execute(const std::vector<Tensor> &outputs, int peer,
                          infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        GroupedRecv, outputs, peer, communicator);
}

void send(const Tensor &input, int peer, infinicclComm_t communicator) {
    Send::execute(input, peer, communicator);
}

void recv_(Tensor output, int peer, infinicclComm_t communicator) {
    Recv::execute(output, peer, communicator);
}

namespace {
template <typename Enqueue>
void run_grouped(infinicclComm_t communicator, Enqueue &&enqueue) {
    // Graph recording retains each P2P operation as a host replay step. The
    // short decode tensors already use that validated segmented path.
    if (infinicore::context::isGraphRecording()) {
        enqueue();
        return;
    }

    INFINICORE_CHECK_ERROR(infinicclGroupStart(communicator));
    try {
        enqueue();
    } catch (...) {
        (void)infinicclGroupEnd(communicator);
        throw;
    }
    INFINICORE_CHECK_ERROR(infinicclGroupEnd(communicator));
}
} // namespace

void send_grouped(
    const std::vector<Tensor> &inputs,
    int peer, infinicclComm_t communicator) {
    if (infinicore::context::isGraphRecording()) {
        GroupedSend::execute(inputs, peer, communicator);
        return;
    }
    run_grouped(communicator, [&] {
        for (const auto &input : inputs) {
            Send::execute(input, peer, communicator);
        }
    });
}

void recv_grouped_(
    const std::vector<Tensor> &outputs,
    int peer, infinicclComm_t communicator) {
    if (infinicore::context::isGraphRecording()) {
        GroupedRecv::execute(outputs, peer, communicator);
        return;
    }
    run_grouped(communicator, [&] {
        for (const auto &output : outputs) {
            Recv::execute(output, peer, communicator);
        }
    });
}
} // namespace infinicore::op::distributed
