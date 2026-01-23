#ifndef __SWIGLU_CUDA_INFO_H__
#define __SWIGLU_CUDA_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::swiglu_cuda {

class SwiGLUCudaInfo {
    SwiGLUCudaInfo() = default;

public:
    infiniDtype_t dtype;
    size_t length;
    size_t ndim;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> c_strides;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;

    static utils::Result<SwiGLUCudaInfo> createSwiGLUCudaInfo(infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc) {
        auto dtype = c_desc->dtype();
        if (dtype != a_desc->dtype() || dtype != b_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        auto shape = c_desc->shape();
        CHECK_SAME_SHAPE(shape, a_desc->shape(), b_desc->shape());

        auto ndim = c_desc->ndim();

        size_t length = 1;
        for (int i = 0; i < (int)ndim; i++) {
            length *= shape[i];
        }

        return utils::Result<SwiGLUCudaInfo>(SwiGLUCudaInfo{
            dtype,
            length,
            ndim,
            shape,
            c_desc->strides(),
            a_desc->strides(),
            b_desc->strides()});
    }
};

} // namespace op::swiglu_cuda

#endif // __SWIGLU_CUDA_INFO_H__
