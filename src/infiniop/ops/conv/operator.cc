#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/conv.h"

#ifdef ENABLE_CPU_API
#include "cpu/conv_cpu.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/conv_cuda.cuh"
#endif

__C __export infiniStatus_t infiniopCreateConvDescriptor(infiniopHandle_t handle,
    infiniopConvDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t w,
    void *pads,
    void *strides,
    void *dilations,
    size_t n){
#define CREATE(CASE, NAMESPACE)                                            \
        case CASE:                                                             \
            return op::conv::NAMESPACE::Descriptor::create(                     \
                handle,                                                        \
                reinterpret_cast<op::conv::NAMESPACE::Descriptor **>(desc_ptr), \
                y_desc,                                                        \
                {x_desc,                                                       \
                 w_desc})

    }