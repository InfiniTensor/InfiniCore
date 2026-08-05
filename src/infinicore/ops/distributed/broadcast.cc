#include "infinicore/ops/distributed/broadcast.hpp"
#include "../../utils.hpp"

#include "infinicore/context/context.hpp"

namespace infinicore::op::distributed {

struct BroadcastPlannedMeta {
    graph::GraphTensor output, input;
    int root;
    infinicclComm_t communicator;
};

Broadcast::Broadcast(Tensor output, const Tensor &input, int root, infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(output->numel() == input->numel());
    INFINICORE_ASSERT(input->numel() > 0);
    planned_meta_ = new BroadcastPlannedMeta{graph::GraphTensor(output), graph::GraphTensor(input), root, communicator};
}

Broadcast::~Broadcast() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<BroadcastPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void Broadcast::run() const {
    auto *meta = reinterpret_cast<BroadcastPlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclBroadcast(meta->input->data(),
                                              meta->output->data(),
                                              meta->input->numel(),
                                              static_cast<infiniDtype_t>(static_cast<int>(meta->input->dtype())),
                                              meta->root,
                                              meta->communicator,
                                              infinicore::context::getStream()));
}

void Broadcast::execute(Tensor output, const Tensor &input, int root, infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Broadcast, output, input, root, communicator);
}

Tensor broadcast(const Tensor &input, int root, infinicclComm_t communicator) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    broadcast_(output, input, root, communicator);
    return output;
}

void broadcast_(Tensor output, const Tensor &input, int root, infinicclComm_t communicator) {
    Broadcast::execute(output, input, root, communicator);
}

} // namespace infinicore::op::distributed
