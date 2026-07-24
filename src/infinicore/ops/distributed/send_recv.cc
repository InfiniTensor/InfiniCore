#include "infinicore/ops/distributed/send_recv.hpp"
#include "../../utils.hpp"

#include "infinicore/context/context.hpp"

namespace infinicore::op::distributed {

struct SendPlannedMeta {
    graph::GraphTensor input;
    int peer;
    infinicclComm_t communicator;
};

struct RecvPlannedMeta {
    graph::GraphTensor output;
    int peer;
    infinicclComm_t communicator;
};

Send::Send(const Tensor &input, int peer, infinicclComm_t communicator) {
    INFINICORE_ASSERT(input->is_contiguous());
    INFINICORE_ASSERT(input->numel() > 0);
    planned_meta_ = new SendPlannedMeta{graph::GraphTensor(input), peer, communicator};
}

Send::~Send() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<SendPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void Send::run() const {
    auto *meta = reinterpret_cast<SendPlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclSend(meta->input->data(),
                                         meta->input->numel(),
                                         static_cast<infiniDtype_t>(static_cast<int>(meta->input->dtype())),
                                         meta->peer,
                                         meta->communicator,
                                         infinicore::context::getStream()));
}

void Send::execute(const Tensor &input, int peer, infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Send, input, peer, communicator);
}

Recv::Recv(Tensor output, int peer, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->is_contiguous());
    INFINICORE_ASSERT(output->numel() > 0);
    planned_meta_ = new RecvPlannedMeta{graph::GraphTensor(output), peer, communicator};
}

Recv::~Recv() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<RecvPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void Recv::run() const {
    auto *meta = reinterpret_cast<RecvPlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclRecv(meta->output->data(),
                                         meta->output->numel(),
                                         static_cast<infiniDtype_t>(static_cast<int>(meta->output->dtype())),
                                         meta->peer,
                                         meta->communicator,
                                         infinicore::context::getStream()));
}

void Recv::execute(Tensor output, int peer, infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Recv, output, peer, communicator);
}

void send(const Tensor &input, int peer, infinicclComm_t communicator) {
    Send::execute(input, peer, communicator);
}

void recv_(Tensor output, int peer, infinicclComm_t communicator) {
    Recv::execute(output, peer, communicator);
}

Tensor recv(const Shape &shape, DataType dtype, Device device, int peer, infinicclComm_t communicator) {
    auto output = Tensor::empty(shape, dtype, device);
    recv_(output, peer, communicator);
    return output;
}

} // namespace infinicore::op::distributed
