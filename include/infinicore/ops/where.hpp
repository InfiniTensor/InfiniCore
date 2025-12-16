#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Where {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor out, Tensor cond, Tensor x, Tensor y);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor where(Tensor cond, Tensor x, Tensor y);
void where_(Tensor out, Tensor cond, Tensor x, Tensor y);

class WhereIndices {
public:
    using schema = std::vector<Tensor> (*)(Tensor);
    static std::vector<Tensor> execute(Tensor cond);
    static common::OpDispatcher<schema> &dispatcher();
};

// where(cond) -> tuple of index tensors
std::vector<Tensor> where_indices(Tensor cond);

} // namespace infinicore::op


