#include "infinicore/ops/moe_sum_vendor.hpp"
#include "../../utils.hpp"
#include "../vendor_ops/vendor_ops_dispatch.hpp"
#include <stdexcept>

namespace infinicore::op {

void moe_sum_vendor_(Tensor output, const Tensor &input, std::optional<Tensor> topk_weights, std::optional<Tensor> extra_residual, double routed_scale, double residual_scale) {
    if (topk_weights && extra_residual) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, *topk_weights, *extra_residual);
    } else if (topk_weights) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, *topk_weights);
    } else if (extra_residual) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, *extra_residual);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    }
    if (input->ndim() != 3 || output->ndim() != 2) {
        throw std::runtime_error("moe_sum_vendor expects input 3D and output 2D");
    }
    if (input->dtype() != DataType::F16 && input->dtype() != DataType::BF16) {
        throw std::runtime_error("moe_sum_vendor expects fp16/bfloat16 input");
    }
    if (output->dtype() != input->dtype()) {
        throw std::runtime_error("moe_sum_vendor output dtype must match input");
    }
    if (!input->is_contiguous() || !output->is_contiguous() || (topk_weights && !(*topk_weights)->is_contiguous()) || (extra_residual && !(*extra_residual)->is_contiguous())) {
        throw std::runtime_error("moe_sum_vendor expects contiguous tensors");
    }
    const size_t n = input->size(0), t = input->size(1), h = input->size(2);
    if (output->size(0) != n || output->size(1) != h) {
        throw std::runtime_error("moe_sum_vendor output shape mismatch");
    }
    if (h == 0 || (h % 2) != 0 || h > 16384) {
        throw std::runtime_error("moe_sum_vendor requires 0 < H <= 16384 and H % 2 == 0");
    }
    if (topk_weights && ((*topk_weights)->dtype() != DataType::F32 || (*topk_weights)->ndim() != 2 || (*topk_weights)->size(0) != n || (*topk_weights)->size(1) != t)) {
        throw std::runtime_error("moe_sum_vendor topk_weights must be float32 [N,T]");
    }
    if (extra_residual && ((*extra_residual)->dtype() != output->dtype() || (*extra_residual)->ndim() != 2 || (*extra_residual)->size(0) != n || (*extra_residual)->size(1) != h)) {
        throw std::runtime_error("moe_sum_vendor extra_residual shape/dtype mismatch");
    }
    auto kernel = vendor_ops::lookup(
        vendor_ops::moe_sum_dispatcher(),
        output->device().getType(),
        "moe_sum_vendor");
    kernel(output, input, topk_weights, extra_residual, routed_scale, residual_scale);
}

} // namespace infinicore::op
