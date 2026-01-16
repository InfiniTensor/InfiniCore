#pragma once

#include "common/op.hpp"

namespace infinicore::op {

Tensor embedding(Tensor input, Tensor weight);
void embedding_(Tensor out, Tensor input, Tensor weight);
} // namespace infinicore::op
