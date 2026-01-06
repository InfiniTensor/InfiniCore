#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class FlashAttention {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float, bool);
    static void execute(Tensor out, Tensor q, Tensor k, Tensor v, float scale, bool is_causal);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor flash_attention(Tensor q, Tensor k, Tensor v, float scale, bool is_causal);
void flash_attention_(Tensor out, Tensor q, Tensor k, Tensor v, float scale, bool is_causal);
} // namespace infinicore::op
