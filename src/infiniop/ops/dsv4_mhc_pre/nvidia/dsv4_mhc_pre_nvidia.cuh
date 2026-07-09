#ifndef DSV4_MHC_PRE_NVIDIA_CUH
#define DSV4_MHC_PRE_NVIDIA_CUH

#include "../dsv4_mhc_pre.h"

namespace op::dsv4_mhc_pre::nvidia {

class Descriptor final : public op::dsv4_mhc_pre::Descriptor {
public:
    using op::dsv4_mhc_pre::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t base_desc, float eps);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *output, const void *input, const void *scale, const void *base, void *stream) const;
};

} // namespace op::dsv4_mhc_pre::nvidia

#endif
