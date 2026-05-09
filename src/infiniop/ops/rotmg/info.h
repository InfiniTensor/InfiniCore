#ifndef __ROTMG_INFO_H__
#define __ROTMG_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

class RotmgInfo {
private:
    RotmgInfo() = default;

public:
    infiniDtype_t data_type;

    static utils::Result<RotmgInfo> createRotmgInfo(
        infiniopTensorDescriptor_t d1_desc,
        infiniopTensorDescriptor_t d2_desc,
        infiniopTensorDescriptor_t x1_desc,
        infiniopTensorDescriptor_t y1_desc,
        infiniopTensorDescriptor_t param_desc) {

        CHECK_OR_RETURN(d1_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(d2_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x1_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y1_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(param_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto data_type = d1_desc->dtype();

        CHECK_OR_RETURN(d2_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(x1_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(y1_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(param_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(param_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(d1_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(d2_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x1_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y1_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(param_desc->numel() == 5, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(param_desc->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<RotmgInfo>(RotmgInfo{
            data_type});
    }
};

#endif // __ROTMG_INFO_H__
