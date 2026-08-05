#pragma once
#include "../tensor.hpp"
#include <cstdint>
namespace infinicore::op {
void vocab_parallel_embedding_(Tensor output, const Tensor &indices, const Tensor &weight, int64_t vocab_start, int64_t vocab_end);
}
