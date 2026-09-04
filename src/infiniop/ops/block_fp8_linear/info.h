#ifndef __BLOCK_FP8_LINEAR_INFO_H__
#define __BLOCK_FP8_LINEAR_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::block_fp8_linear {

static constexpr size_t BLOCK_SIZE = 128;

class BlockFP8LinearInfo {
    BlockFP8LinearInfo() = default;

public:
    infiniDtype_t dtype;  // activation/output dtype (BF16)
    size_t M;
    size_t N;
    size_t K;
    size_t num_out_blocks;  // ceil(N/128)
    size_t num_in_blocks;   // ceil(K/128)

    static utils::Result<BlockFP8LinearInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t weight_scale_desc) {
        CHECK_OR_RETURN(output_desc != nullptr && input_desc != nullptr
                            && weight_desc != nullptr && weight_scale_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);

        const auto dtype = input_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_BF16);
        CHECK_OR_RETURN(output_desc->dtype() == dtype,
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(weight_desc->dtype(), INFINI_DTYPE_F8);
        CHECK_DTYPE(weight_scale_desc->dtype(), INFINI_DTYPE_F32);

        CHECK_OR_RETURN(input_desc->ndim() == 2
                            && output_desc->ndim() == 2
                            && weight_desc->ndim() == 2
                            && weight_scale_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(input_desc->isContiguous() && output_desc->isContiguous()
                            && weight_desc->isContiguous()
                            && weight_scale_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        const size_t M = input_desc->dim(0);
        const size_t K = input_desc->dim(1);
        const size_t N = weight_desc->dim(0);
        const size_t K_w = weight_desc->dim(1);

        CHECK_OR_RETURN(K == K_w && K > 0 && N > 0 && M > 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->dim(0) == M && output_desc->dim(1) == N,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        const size_t num_out_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const size_t num_in_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CHECK_OR_RETURN(weight_scale_desc->dim(0) == num_out_blocks
                            && weight_scale_desc->dim(1) == num_in_blocks,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        BlockFP8LinearInfo info;
        info.dtype = dtype;
        info.M = M;
        info.N = N;
        info.K = K;
        info.num_out_blocks = num_out_blocks;
        info.num_in_blocks = num_in_blocks;
        return utils::Result<BlockFP8LinearInfo>(info);
    }
};

} // namespace op::block_fp8_linear

#endif
