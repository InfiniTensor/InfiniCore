#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <infiniccl.h>

namespace infinicore::op::distributed {

class Broadcast : public graph::GraphOperator {
public:
    Broadcast(Tensor output, const Tensor &input, int root, infinicclComm_t communicator);
    ~Broadcast();
    void run() const override;
    static void execute(Tensor output, const Tensor &input, int root, infinicclComm_t communicator);

private:
    void *planned_meta_;
};

Tensor broadcast(const Tensor &input, int root, infinicclComm_t communicator);
void broadcast_(Tensor output, const Tensor &input, int root, infinicclComm_t communicator);

} // namespace infinicore::op::distributed
