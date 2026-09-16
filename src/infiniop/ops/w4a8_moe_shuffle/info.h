#ifndef __W4A8_MOE_SHUFFLE_INFO_H__
#define __W4A8_MOE_SHUFFLE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::w4a8_moe_shuffle {

class W4A8MoeShuffleInfo {
    W4A8MoeShuffleInfo() = default;

public:
    size_t output_size;
    size_t packed_k;
    size_t n;
    size_t n_tile;

    static utils::Result<W4A8MoeShuffleInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc) {
        CHECK_OR_RETURN(output_desc != nullptr && input_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(output_desc->dtype() == INFINI_DTYPE_I8
                            && input_desc->dtype() == INFINI_DTYPE_I8,
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(output_desc->ndim() == 2 && input_desc->ndim() == 2
                            && output_desc->isContiguous()
                            && input_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->dim(0) == input_desc->dim(0)
                            && output_desc->dim(1) == input_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        const size_t n = input_desc->dim(0);
        const size_t packed_k = input_desc->dim(1);
        const size_t n_tile = n % 256 == 0 ? 256 : n;
        CHECK_OR_RETURN(n % 32 == 0 && packed_k % 32 == 0
                            && n_tile % 32 == 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        return utils::Result<W4A8MoeShuffleInfo>(
            W4A8MoeShuffleInfo{n * packed_k, packed_k, n, n_tile});
    }
};

} // namespace op::w4a8_moe_shuffle

#endif
