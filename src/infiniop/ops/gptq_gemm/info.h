#ifndef __GPTQ_GEMM_INFO_H__
#define __GPTQ_GEMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <optional>
#include <vector>

namespace op::gptq_gemm {

class GptqGemmInfo {
    GptqGemmInfo() = default;

public:
    // --- Data Type ---
    infiniDtype_t dtype;

    // --- Shape Dimensions ---
    size_t M, K, N, b_size_0;
    int block_size, num_groups;
    bool use_exllama;
    int64_t quant_bit;

    static utils::Result<GptqGemmInfo> createGptqGemmInfo(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t b_scales_desc,
        infiniopTensorDescriptor_t b_zeros_desc,
        infiniopTensorDescriptor_t b_g_idx_desc,
        bool use_exllama,
        int quant_bit) {

        auto dtype = out_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16);
        if (b_scales_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (b_zeros_desc->dtype() != INFINI_DTYPE_I32 || b_g_idx_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        size_t M = out_desc->shape()[0];
        size_t N = out_desc->shape()[1];
        size_t K = a_desc->shape()[1];
        size_t b_size_0 = b_desc->shape()[0];
        int block_size = 128;
        int num_groups = K / block_size;
        if (quant_bit != 4) {
            throw std::runtime_error(
                "quant_bit must be 4, but got " + std::to_string(quant_bit));
        }

        auto ndim = out_desc->ndim();
        CHECK_OR_RETURN(ndim == 2
                            && a_desc->ndim() == ndim
                            && b_desc->ndim() == ndim
                            && b_scales_desc->ndim() == ndim
                            && b_zeros_desc->ndim() == ndim,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(b_scales_desc->shape()[1] == N
                            && b_scales_desc->shape()[0] == num_groups
                            && b_zeros_desc->shape()[1] == N
                            && b_zeros_desc->shape()[0] == num_groups,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<GptqGemmInfo>(GptqGemmInfo{
            dtype,
            M, K, N, b_size_0,
            block_size, num_groups,
            use_exllama, static_cast<int64_t>(quant_bit)});
    }
};

} // namespace op::gptq_gemm

#endif // __GPTQ_GEMM_INFO_H__
