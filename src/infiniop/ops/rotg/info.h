#ifndef __ROTG_INFO_H__
#define __ROTG_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

class RotgInfo {
private:
    RotgInfo() = default;

public:
    infiniDtype_t data_type;

    static utils::Result<RotgInfo> createRotgInfo(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t s_desc) {

        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(c_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(s_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto data_type = x_desc->dtype();

        CHECK_OR_RETURN(y_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(c_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(s_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(x_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(c_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(s_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<RotgInfo>(RotgInfo{
            data_type});
    }
};

#endif // __ROTG_INFO_H__
