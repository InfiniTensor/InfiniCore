#include "infinicore/ops/scaled_mm_w8a8.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

namespace infinicore::op {
Tensor scaled_mm_w8a8(const Tensor &a, const Tensor &b, const Tensor &a_scales,
                      const Tensor &b_scales, std::optional<Tensor> bias, bool trans_weight) {
    if (a->ndim() != 2 || b->ndim() != 2) {
        throw std::runtime_error("scaled_mm_w8a8 expects 2D a and b");
    }
    const size_t n = trans_weight ? b->size(0) : b->size(1);
    Tensor out = Tensor::empty({a->size(0), n}, bias ? (*bias)->dtype() : DataType::F16, a->device());
    scaled_mm_w8a8_(out, a, b, a_scales, b_scales, bias, trans_weight);
    return out;
}

void scaled_mm_w8a8_(Tensor out, const Tensor &a, const Tensor &b, const Tensor &a_scales,
                     const Tensor &b_scales, std::optional<Tensor> bias, bool trans_weight) {
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b, a_scales, b_scales, *bias);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b, a_scales, b_scales);
    }
    if (a->ndim() != 2 || b->ndim() != 2 || out->ndim() != 2 || a_scales->ndim() != 2 || b_scales->ndim() != 2) {
        throw std::runtime_error("scaled_mm_w8a8 expects 2D tensors");
    }
    if (a->dtype() != DataType::I8 || b->dtype() != DataType::I8) {
        throw std::runtime_error("scaled_mm_w8a8 expects int8 a and b");
    }
    if (a_scales->dtype() != DataType::F32 || b_scales->dtype() != DataType::F32) {
        throw std::runtime_error("scaled_mm_w8a8 expects float32 scales");
    }
    if (out->dtype() != DataType::F16 && out->dtype() != DataType::BF16) {
        throw std::runtime_error("scaled_mm_w8a8 expects fp16/bfloat16 out");
    }
    const size_t k = a->size(1);
    const size_t n = trans_weight ? b->size(0) : b->size(1);
    if ((!trans_weight && b->size(0) != k) || (trans_weight && b->size(1) != k)) {
        throw std::runtime_error("scaled_mm_w8a8 K dimension mismatch");
    }
    if (out->size(0) != a->size(0) || out->size(1) != n) {
        throw std::runtime_error("scaled_mm_w8a8 out shape mismatch");
    }
    if (a_scales->size(0) != a->size(0) || a_scales->size(1) != 1) {
        throw std::runtime_error("scaled_mm_w8a8 expects a_scales (M,1)");
    }
    if (b_scales->size(0) != n || b_scales->size(1) != 1) {
        throw std::runtime_error("scaled_mm_w8a8 expects b_scales (N,1)");
    }
    if (bias && ((*bias)->ndim() != 1 || (*bias)->dtype() != out->dtype() || (*bias)->numel() != n)) {
        throw std::runtime_error("scaled_mm_w8a8 invalid bias");
    }
    if (!out->is_contiguous() || !a->is_contiguous() || !b->is_contiguous() || !a_scales->is_contiguous() || !b_scales->is_contiguous() || (bias && !(*bias)->is_contiguous())) {
        throw std::runtime_error("scaled_mm_w8a8 expects contiguous tensors");
    }
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (out->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::scaled_mm_w8a8_available()) {
            throw std::runtime_error("scaled_mm_w8a8 requires vllm_iluvatar cuinfer extension");
        }
        auto o = adaptor::to_aten_tensor(out);
        auto aa = adaptor::to_aten_tensor(a);
        auto bb = adaptor::to_aten_tensor(b);
        auto as = adaptor::to_aten_tensor(a_scales);
        auto bs = adaptor::to_aten_tensor(b_scales);
        std::optional<at::Tensor> bi;
        if (bias) {
            bi = adaptor::to_aten_tensor(*bias);
        }
        adaptor::vllm_iluvatar::scaled_mm_w8a8(o, aa, bb, as, bs, bi, trans_weight);
        return;
    }
#endif
    throw std::runtime_error("scaled_mm_w8a8 currently supports only Iluvatar with vllm_iluvatar");
}
} // namespace infinicore::op
