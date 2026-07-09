#ifndef DSV4_SGLANG_MEGA_MOE_PRE_DISPATCH_NVIDIA_CUH
#define DSV4_SGLANG_MEGA_MOE_PRE_DISPATCH_NVIDIA_CUH
#include "../dsv4_sglang_mega_moe_pre_dispatch.h"
namespace op::dsv4_sglang_mega_moe_pre_dispatch::nvidia {
class Descriptor final : public op::dsv4_sglang_mega_moe_pre_dispatch::Descriptor {
public:
    using op::dsv4_sglang_mega_moe_pre_dispatch::Descriptor::Descriptor;
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t topk_idx_desc, infiniopTensorDescriptor_t topk_weights_desc, infiniopTensorDescriptor_t buf_x_desc, infiniopTensorDescriptor_t buf_x_sf_desc, infiniopTensorDescriptor_t buf_topk_idx_desc, infiniopTensorDescriptor_t buf_topk_weights_desc);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, const void *x, const void *topk_idx, const void *topk_weights, void *buf_x, void *buf_x_sf, void *buf_topk_idx, void *buf_topk_weights, void *stream) const;
};
} // namespace op::dsv4_sglang_mega_moe_pre_dispatch::nvidia
#endif
