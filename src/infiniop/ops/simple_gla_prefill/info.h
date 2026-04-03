#ifndef __SIMPLE_GLA_PREFILL_CUDA_INFO_H__
#define __SIMPLE_GLA_PREFILL_CUDA_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::simple_gla_prefill_cuda {

class SimpleGLAPrefillCudaInfo {
    SimpleGLAPrefillCudaInfo() = default;

public:
    infiniDtype_t dtype;
    size_t B, T, H, D;

    static utils::Result<SimpleGLAPrefillCudaInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t g_gamma_desc) {

        auto dtype = out_desc->dtype();
        if (dtype != q_desc->dtype() || dtype != k_desc->dtype() || dtype != v_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        // Only support half/bf16 for now.
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

        // Shapes must match and be 4D [B,T,H,D]
        const auto &out_shape = out_desc->shape();
        CHECK_SAME_SHAPE(out_shape, q_desc->shape(), k_desc->shape(), v_desc->shape());
        if (out_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        // g_gamma: [H], F32
        if (g_gamma_desc->ndim() != 1 || g_gamma_desc->shape()[0] != out_shape[2]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (g_gamma_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        return utils::Result<SimpleGLAPrefillCudaInfo>(SimpleGLAPrefillCudaInfo{
            dtype,
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3],
        });
    }
};

} // namespace op::simple_gla_prefill_cuda

#endif // __SIMPLE_GLA_PREFILL_CUDA_INFO_H__

