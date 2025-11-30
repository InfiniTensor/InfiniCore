#pragma once

#include "common/op.hpp"

namespace infinicore::op {

Tensor cat(std::vector<Tensor> tensors, int dim);
void cat_(std::vector<Tensor> tensors, int dim, Tensor out);
} // namespace infinicore::op