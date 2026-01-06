#include "infinicore/ops/flash_attention.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<FlashAttention::schema> &FlashAttention::dispatcher() {
    static common::OpDispatcher<FlashAttention::schema> dispatcher_;
    return dispatcher_;
};

void FlashAttention::execute(Tensor out, Tensor q, Tensor k, Tensor v, float scale, bool is_causal) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v);
    infinicore::context::setDevice(out->device());
    dispatcher().lookup(out->device().getType())(
        out, q, k, v, scale, is_causal);
}

Tensor flash_attention(Tensor q, Tensor k, Tensor v, float scale, bool is_causal) {
    Shape shape = q->shape();
    auto out = Tensor::empty(shape, q->dtype(), q->device());
    flash_attention_(out, q, k, v, scale, is_causal);
    return out;
}

void flash_attention_(Tensor out, Tensor q, Tensor k, Tensor v, float scale, bool is_causal) {
    FlashAttention::execute(out, q, k, v, scale, is_causal);
}
} // namespace infinicore::op
