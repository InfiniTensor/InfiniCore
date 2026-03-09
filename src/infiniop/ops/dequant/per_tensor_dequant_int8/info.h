#ifndef __PER_TENSOR_DEQUANT_INT8_INFO_H__
#define __PER_TENSOR_DEQUANT_INT8_INFO_H__

#include "../../../../utils.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::per_tensor_dequant_int8 {

class PerTensorDequantI8Info {
private:
    PerTensorDequantI8Info() = default;

public:
    infiniDtype_t dtype, packed_type;
    int num_elements;

    static utils::Result<PerTensorDequantI8Info> createPerTensorDequantI8Info(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t x_packed_desc,
        infiniopTensorDescriptor_t x_scale_desc,
        infiniopTensorDescriptor_t x_zero_desc) {

        CHECK_OR_RETURN(
            x_packed_desc != nullptr && x_scale_desc != nullptr && x_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t dtype = x_desc->dtype();
        const infiniDtype_t packed_type = x_packed_desc->dtype();

        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE(packed_type, INFINI_DTYPE_I8);

        CHECK_OR_RETURN(x_desc->ndim() == 2
                            && x_packed_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto shape = x_desc->shape();
        CHECK_SAME_SHAPE(shape, x_packed_desc->shape());

        auto ndim = x_desc->ndim();

        int num_elements = 1;
        for (int i = 0; i < (int)ndim; i++) {
            num_elements *= static_cast<int>(shape[i]);
        }

        return utils::Result<PerTensorDequantI8Info>(PerTensorDequantI8Info{
            dtype,
            packed_type,
            num_elements});
    }
};

} // namespace op::per_tensor_dequant_int8

#endif //  __PER_TENSOR_DEQUANT_INT8_INFO_H__
