#include "addr_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <spdlog/spdlog.h>
namespace op::addr::cpu {
Descriptor::~Descriptor() = default;

// Template function to handle different data types
template <typename Tdata>
infiniStatus_t addr_impl(Tdata *out,
                         const Tdata *input,
                         const Tdata *vec1,
                         const Tdata *vec2,
                         const AddrInfo &info,
                         void *workspace,
                         size_t workspace_size) {
    size_t n = info.vec1_size;
    size_t m = info.vec2_size;
    float beta = info.beta;
    float alpha = info.alpha;

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
                float a = utils::cast<float>(vec1[i * info.vec1_stride]);
                float b = utils::cast<float>(vec2[j * info.vec2_stride]);
                float c = utils::cast<float>(input[i * info.input_stride0 + j * info.input_stride1]);
                out[i * info.output_stride0 + j * info.output_stride1] = utils::cast<Tdata>(alpha * a * b + beta * c);
            } else {
                float a = vec1[i * info.vec1_stride], b = vec2[j * info.vec2_stride], c = input[i * info.input_stride0 + j * info.input_stride1];
                out[i * info.output_stride0 + j * info.output_stride1] = utils::cast<Tdata>(alpha * a * b + beta * c);
            }
        }
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t vec1_desc,
    infiniopTensorDescriptor_t vec2_desc,
    float beta,
    float alpha) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = AddrInfo::create(input_desc, out_desc, vec1_desc, vec2_desc, beta, alpha);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(info.take(), 0, nullptr,
                               INFINI_DEVICE_CPU, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *input,
    const void *vec1,
    const void *vec2,
    void *stream) const {

    switch (_info.dtype) {
    case INFINI_DTYPE_F32:
        return addr_impl(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(input),
                         reinterpret_cast<const float *>(vec1),
                         reinterpret_cast<const float *>(vec2),
                         _info, workspace, workspace_size);
        break;
    case INFINI_DTYPE_F16:
        return addr_impl(reinterpret_cast<fp16_t *>(out),
                         reinterpret_cast<const fp16_t *>(input),
                         reinterpret_cast<const fp16_t *>(vec1),
                         reinterpret_cast<const fp16_t *>(vec2),
                         _info, workspace, workspace_size);
    case INFINI_DTYPE_BF16:
        return addr_impl(reinterpret_cast<bf16_t *>(out),
                         reinterpret_cast<const bf16_t *>(input),
                         reinterpret_cast<const bf16_t *>(vec1),
                         reinterpret_cast<const bf16_t *>(vec2),
                         _info, workspace, workspace_size);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::addr::cpu