#include "infinicore/ops/scaled_mm_w4a8.hpp"
#include "../../utils.hpp"
#include "../vendor_ops/vendor_ops_dispatch.hpp"
#include <stdexcept>

namespace infinicore::op {
Tensor scaled_mm_w4a8(const Tensor &a, const Tensor &b, const Tensor &a_scales, const Tensor &b_scales, std::optional<Tensor> bias, bool trans_weight) {
    if (a->ndim() != 2 || b->ndim() != 2) {
        throw std::runtime_error("scaled_mm_w4a8 expects 2D a and b");
    }
    const size_t m = a->size(0);
    const size_t n = trans_weight ? b->size(0) : b->size(1) * 2;
    Tensor out = Tensor::empty({m, n}, bias ? (*bias)->dtype() : DataType::F16, a->device());
    scaled_mm_w4a8_(out, a, b, a_scales, b_scales, bias, trans_weight);
    return out;
}

void scaled_mm_w4a8_(Tensor out, const Tensor &a, const Tensor &b, const Tensor &a_scales, const Tensor &b_scales, std::optional<Tensor> bias, bool trans_weight) {
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b, a_scales, b_scales, *bias);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b, a_scales, b_scales);
    }
    if (a->ndim() != 2 || b->ndim() != 2 || out->ndim() != 2 || a_scales->ndim() != 2 || b_scales->ndim() != 2) {
        throw std::runtime_error("scaled_mm_w4a8 expects 2D tensors");
    }
    if (a->dtype() != DataType::I8 || b->dtype() != DataType::I8) {
        throw std::runtime_error("scaled_mm_w4a8 expects int8 a and packed int8 b");
    }
    if (a_scales->dtype() != DataType::F32 || b_scales->dtype() != DataType::F32) {
        throw std::runtime_error("scaled_mm_w4a8 expects float32 scales");
    }
    if (out->dtype() != DataType::F16 && out->dtype() != DataType::BF16) {
        throw std::runtime_error("scaled_mm_w4a8 expects fp16/bfloat16 out");
    }
    if (bias && ((*bias)->ndim() != 1 || (*bias)->dtype() != out->dtype() || (*bias)->numel() != out->size(1))) {
        throw std::runtime_error("scaled_mm_w4a8 expects bias shape (N,) and same dtype as out");
    }
    if (!out->is_contiguous() || !a->is_contiguous() || !b->is_contiguous() || !a_scales->is_contiguous() || !b_scales->is_contiguous() || (bias && !(*bias)->is_contiguous())) {
        throw std::runtime_error("scaled_mm_w4a8 expects contiguous tensors");
    }
    const size_t k = a->size(1);
    if ((!trans_weight && b->size(0) != k) || (trans_weight && b->size(1) * 2 != k)) {
        throw std::runtime_error("scaled_mm_w4a8 K dimension mismatch");
    }
    const size_t n = trans_weight ? b->size(0) : b->size(1) * 2;
    if (out->size(0) != a->size(0) || out->size(1) != n) {
        throw std::runtime_error("scaled_mm_w4a8 out shape mismatch");
    }
    if (a_scales->size(0) != a->size(0) || a_scales->size(1) != 1) {
        throw std::runtime_error("scaled_mm_w4a8 expects a_scales shape (M,1)");
    }
    if (b_scales->size(0) != out->size(1) || b_scales->size(1) != 1) {
        throw std::runtime_error("scaled_mm_w4a8 expects b_scales shape (N,1)");
    }
    auto kernel = vendor_ops::lookup(
        vendor_ops::scaled_mm_w4a8_dispatcher(),
        out->device().getType(),
        "scaled_mm_w4a8");
    kernel(out, a, b, a_scales, b_scales, bias, trans_weight);
}
} // namespace infinicore::op
