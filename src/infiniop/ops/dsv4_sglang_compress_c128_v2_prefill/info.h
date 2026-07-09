#ifndef DSV4_SGLANG_COMPRESS_C128_V2_PREFILL_INFO_H
#define DSV4_SGLANG_COMPRESS_C128_V2_PREFILL_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>
namespace op::dsv4_sglang_compress_c128_v2_prefill {
struct TensorInfo {
    infiniDtype_t dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};
struct Info {
    std::vector<TensorInfo> tensors;
};
inline TensorInfo makeTensorInfo(infiniopTensorDescriptor_t desc) {
    TensorInfo info;
    info.dtype = desc->dtype();
    for (size_t i = 0; i < desc->ndim(); ++i) {
        info.shape.push_back(static_cast<int64_t>(desc->dim(i)));
        info.strides.push_back(static_cast<int64_t>(desc->stride(i)));
    }
    return info;
}
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t kv_buffer_desc, infiniopTensorDescriptor_t kv_input_desc, infiniopTensorDescriptor_t kv_output_desc, infiniopTensorDescriptor_t ape_desc, infiniopTensorDescriptor_t plan_c_desc, infiniopTensorDescriptor_t plan_w_desc) {
    CHECK_OR_RETURN(info, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(kv_buffer_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(kv_buffer_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(kv_input_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(kv_input_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(kv_output_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(kv_output_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(ape_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(ape_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(plan_c_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(plan_c_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(plan_w_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(plan_w_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{std::vector<TensorInfo>{makeTensorInfo(kv_buffer_desc), makeTensorInfo(kv_input_desc), makeTensorInfo(kv_output_desc), makeTensorInfo(ape_desc), makeTensorInfo(plan_c_desc), makeTensorInfo(plan_w_desc)}};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_sglang_compress_c128_v2_prefill
#endif
