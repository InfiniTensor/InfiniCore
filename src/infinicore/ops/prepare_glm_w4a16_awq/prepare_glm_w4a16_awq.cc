#include "infinicore/ops/prepare_glm_w4a16_awq.hpp"
#include "../../utils.hpp"
#include "../vendor_ops/vendor_ops_dispatch.hpp"
#include <stdexcept>

namespace infinicore::op {
void prepare_glm_w4a16_awq_(Tensor qweight, Tensor qzeros, Tensor scales,
                            const Tensor &checkpoint_weight,
                            const Tensor &channel_scales) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(qweight, qzeros, scales,
                                          checkpoint_weight, channel_scales);
    if (checkpoint_weight->ndim() != 2 || channel_scales->ndim() != 2) {
        throw std::runtime_error("prepare_glm_w4a16_awq expects 2D checkpoint tensors");
    }
    const size_t n = checkpoint_weight->size(0);
    const size_t k = checkpoint_weight->size(1) * 2;
    if ((k % 256) != 0 || (n % 2) != 0
        || qweight->shape() != std::vector<size_t>{k, n / 2}
        || qzeros->shape() != std::vector<size_t>{k / 64, n / 2}
        || scales->shape() != std::vector<size_t>{k / 64, n}) {
        throw std::runtime_error("prepare_glm_w4a16_awq output shape mismatch");
    }
    if (qweight->dtype() != DataType::I8 || qzeros->dtype() != DataType::I8
        || scales->dtype() != DataType::BF16
        || checkpoint_weight->dtype() != DataType::I8
        || channel_scales->dtype() != DataType::F32) {
        throw std::runtime_error("prepare_glm_w4a16_awq dtype mismatch");
    }
    if (!qweight->is_contiguous() || !qzeros->is_contiguous()
        || !scales->is_contiguous() || !checkpoint_weight->is_contiguous()
        || !channel_scales->is_contiguous()) {
        throw std::runtime_error("prepare_glm_w4a16_awq expects contiguous tensors");
    }
    auto kernel = vendor_ops::lookup(
        vendor_ops::prepare_glm_w4a16_awq_dispatcher(),
        qweight->device().getType(), "prepare_glm_w4a16_awq");
    kernel(qweight, qzeros, scales, checkpoint_weight, channel_scales);
}
} // namespace infinicore::op
