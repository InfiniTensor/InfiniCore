#ifndef DSV4_SGLANG_RMSNORM_NVIDIA_CUH
#define DSV4_SGLANG_RMSNORM_NVIDIA_CUH

#include "../dsv4_sglang_rmsnorm.h"

namespace op::dsv4_sglang_rmsnorm::nvidia {

class Descriptor final : public op::dsv4_sglang_rmsnorm::Descriptor {
public:
    using op::dsv4_sglang_rmsnorm::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, double eps);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *output, const void *input, void *stream) const;
};

} // namespace op::dsv4_sglang_rmsnorm::nvidia

#endif
