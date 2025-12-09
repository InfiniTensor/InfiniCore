#pragma once

#include "../device.hpp"
#include "../tensor.hpp"
#include "../nn/rope.hpp"
#include "common/op.hpp"


namespace infinicore::op {
class RoPE {
public:
    using schema = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, RoPEAlgo);
    static void execute(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, RoPEAlgo algo);
    static common::OpDispatcher<schema> &dispatcher();
};

void rope_(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, RoPEAlgo algo);

Tensor rope(const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, RoPEAlgo algo);
} // namespace infinicore::op
