#ifndef __MUL_SCALAR_INFO_H__
#define __MUL_SCALAR_INFO_H__

#include "../../../utils.h"
#include "../../elementwise/elementwise.h"
#include "../../tensor.h"

class MulScalarInfo {
private:
    MulScalarInfo(
        op::elementwise::ElementwiseInfo elementwise_info_,
        infiniDtype_t data_type_,
        bool contiguous_)
        : elementwise_info(std::move(elementwise_info_)),
          data_type(data_type_),
          contiguous(contiguous_) {}

public:
    op::elementwise::ElementwiseInfo elementwise_info;
    infiniDtype_t data_type;
    bool contiguous;

    size_t numel() const { return elementwise_info.getOutputSize(); }

    static utils::Result<MulScalarInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc) {

        CHECK_OR_RETURN(output_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(input_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto data_type = output_desc->dtype();
        CHECK_OR_RETURN(input_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_SAME_SHAPE(output_desc->shape(), input_desc->shape());

        auto elementwise_info_result = op::elementwise::ElementwiseInfo::create(output_desc, {input_desc});
        CHECK_RESULT(elementwise_info_result);

        return utils::Result<MulScalarInfo>(MulScalarInfo(
            elementwise_info_result.take(),
            data_type,
            output_desc->isContiguous() && input_desc->isContiguous()));
    }
};

#endif // __MUL_SCALAR_INFO_H__
