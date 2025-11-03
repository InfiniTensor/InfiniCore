#pragma once

#include "../device.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
#include <infiniop.h>

namespace infinicore::op {
class RoPE {
public:
    using schema = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, infiniopRoPEAlgo_t);
    static void execute(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, infiniopRoPEAlgo_t algo);
    static common::OpDispatcher<schema> &dispatcher();
};

void rope_(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, infiniopRoPEAlgo_t algo);
Tensor rope(const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, infiniopRoPEAlgo_t algo);
} // namespace infinicore::op
