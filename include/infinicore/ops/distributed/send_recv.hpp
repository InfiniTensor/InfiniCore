#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <infiniccl.h>

namespace infinicore::op::distributed {

class Send : public graph::GraphOperator {
public:
    Send(const Tensor &input, int peer, infinicclComm_t communicator);
    ~Send();
    void run() const override;
    static void execute(const Tensor &input, int peer, infinicclComm_t communicator);

private:
    void *planned_meta_;
};

class Recv : public graph::GraphOperator {
public:
    Recv(Tensor output, int peer, infinicclComm_t communicator);
    ~Recv();
    void run() const override;
    static void execute(Tensor output, int peer, infinicclComm_t communicator);

private:
    void *planned_meta_;
};

void send(const Tensor &input, int peer, infinicclComm_t communicator);
void recv_(Tensor output, int peer, infinicclComm_t communicator);
Tensor recv(const Shape &shape, DataType dtype, Device device, int peer, infinicclComm_t communicator);

} // namespace infinicore::op::distributed
