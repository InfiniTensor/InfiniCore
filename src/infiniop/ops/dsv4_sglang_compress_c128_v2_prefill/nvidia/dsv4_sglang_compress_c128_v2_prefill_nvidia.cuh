#ifndef DSV4_SGLANG_COMPRESS_C128_V2_PREFILL_NVIDIA_CUH_H
#define DSV4_SGLANG_COMPRESS_C128_V2_PREFILL_NVIDIA_CUH_H
#include "../dsv4_sglang_compress_c128_v2_prefill.h"
namespace op::dsv4_sglang_compress_c128_v2_prefill::nvidia {
class Descriptor final : public op::dsv4_sglang_compress_c128_v2_prefill::Descriptor {
public:
    using op::dsv4_sglang_compress_c128_v2_prefill::Descriptor::Descriptor;
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t kv_buffer_desc, infiniopTensorDescriptor_t kv_input_desc, infiniopTensorDescriptor_t kv_output_desc, infiniopTensorDescriptor_t ape_desc, infiniopTensorDescriptor_t plan_c_desc, infiniopTensorDescriptor_t plan_w_desc);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, const void *kv_buffer, const void *kv_input, void *kv_output, const void *ape, const void *plan_c, const void *plan_w, void *stream) const;
};
} // namespace op::dsv4_sglang_compress_c128_v2_prefill::nvidia
#endif
