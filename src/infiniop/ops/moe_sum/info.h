#ifndef __MOE_SUM_INFO_H__
#define __MOE_SUM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::moe_sum {

class MoeSumInfo {
    MoeSumInfo() = default;

public:
    infiniDtype_t dtype;
    size_t num_tokens;
    size_t topk;
    size_t hidden_size;

    static utils::Result<MoeSumInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc) {
        auto dtype = input_desc->dtype();
        if (dtype != output_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        if (input_desc->ndim() != 3 || output_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &input_shape = input_desc->shape();
        const auto &output_shape = output_desc->shape();
        const size_t num_tokens = input_shape[0];
        const size_t topk = input_shape[1];
        const size_t hidden_size = input_shape[2];
        if (output_shape[0] != num_tokens || output_shape[1] != hidden_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->strides()[2] != 1 || input_desc->strides()[1] != static_cast<ptrdiff_t>(hidden_size) || output_desc->strides()[1] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<MoeSumInfo>(MoeSumInfo{
            dtype,
            num_tokens,
            topk,
            hidden_size,
        });
    }
};

} // namespace op::moe_sum

#endif
