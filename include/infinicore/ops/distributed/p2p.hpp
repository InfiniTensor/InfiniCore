#pragma once

#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <infiniccl.h>
#include <vector>

namespace infinicore::op::distributed {
class Send : public graph::GraphOperator {
public:
    Send(const Tensor &input, int peer, infinicclComm_t communicator);
    ~Send();
    void run() const override;
    bool is_device_graph_capture_safe() const override {
        return false;
    }
    static void execute(
        const Tensor &input, int peer, infinicclComm_t communicator);

private:
    void *planned_meta_;
};

class Recv : public graph::GraphOperator {
public:
    Recv(Tensor output, int peer, infinicclComm_t communicator);
    ~Recv();
    void run() const override;
    bool is_device_graph_capture_safe() const override {
        return false;
    }
    static void execute(
        Tensor output, int peer, infinicclComm_t communicator);

private:
    void *planned_meta_;
};

class GroupedSend : public graph::GraphOperator {
public:
    GroupedSend(const std::vector<Tensor> &inputs, int peer,
                infinicclComm_t communicator);
    ~GroupedSend();
    void run() const override;
    bool is_device_graph_capture_safe() const override {
        return false;
    }
    static void execute(const std::vector<Tensor> &inputs, int peer,
                        infinicclComm_t communicator);

private:
    void *planned_meta_;
};

class GroupedRecv : public graph::GraphOperator {
public:
    GroupedRecv(const std::vector<Tensor> &outputs, int peer,
                infinicclComm_t communicator);
    ~GroupedRecv();
    void run() const override;
    bool is_device_graph_capture_safe() const override {
        return false;
    }
    static void execute(const std::vector<Tensor> &outputs, int peer,
                        infinicclComm_t communicator);

private:
    void *planned_meta_;
};

void send(const Tensor &input, int peer, infinicclComm_t communicator);
void recv_(Tensor output, int peer, infinicclComm_t communicator);
void send_grouped(
    const std::vector<Tensor> &inputs,
    int peer, infinicclComm_t communicator);
void recv_grouped_(
    const std::vector<Tensor> &outputs,
    int peer, infinicclComm_t communicator);
} // namespace infinicore::op::distributed
