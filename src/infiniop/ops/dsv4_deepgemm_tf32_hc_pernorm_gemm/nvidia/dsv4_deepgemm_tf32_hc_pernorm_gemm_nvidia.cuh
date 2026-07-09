#ifndef DSV4_DEEPGEMM_TF32_HC_PERNORM_GEMM_NVIDIA_CUH
#define DSV4_DEEPGEMM_TF32_HC_PERNORM_GEMM_NVIDIA_CUH

#include "../dsv4_deepgemm_tf32_hc_pernorm_gemm.h"

namespace op::dsv4_deepgemm_tf32_hc_pernorm_gemm::nvidia {

class Descriptor final : public op::dsv4_deepgemm_tf32_hc_pernorm_gemm::Descriptor {
public:
    Descriptor(Info info, size_t workspace_size, infiniDevice_t device_type, int device_id)
        : op::dsv4_deepgemm_tf32_hc_pernorm_gemm::Descriptor(info, workspace_size, device_type, device_id) {}

    static infiniStatus_t create(infiniopHandle_t handle,
                                 Descriptor **desc_ptr,
                                 infiniopTensorDescriptor_t a_desc,
                                 infiniopTensorDescriptor_t b_desc,
                                 infiniopTensorDescriptor_t d_desc,
                                 infiniopTensorDescriptor_t sqr_sum_desc,
                                 int64_t num_splits);

    infiniStatus_t calculate(void *workspace,
                             size_t workspace_size,
                             const void *a,
                             const void *b,
                             void *d,
                             void *sqr_sum,
                             void *stream) const;
};

} // namespace op::dsv4_deepgemm_tf32_hc_pernorm_gemm::nvidia

#endif
