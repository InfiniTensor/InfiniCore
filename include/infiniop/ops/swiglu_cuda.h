#ifndef __INFINIOP_SWIGLU_CUDA_API_H__
#define __INFINIOP_SWIGLU_CUDA_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSwiGLUCudaDescriptor_t;

__C __export infiniStatus_t infiniopCreateSwiGLUCudaDescriptor(infiniopHandle_t handle,
                                                               infiniopSwiGLUCudaDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t c_desc,
                                                               infiniopTensorDescriptor_t a_desc,
                                                               infiniopTensorDescriptor_t b_desc);

__C __export infiniStatus_t infiniopGetSwiGLUCudaWorkspaceSize(infiniopSwiGLUCudaDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSwiGLUCuda(infiniopSwiGLUCudaDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *c,
                                               void const *a,
                                               void const *b,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroySwiGLUCudaDescriptor(infiniopSwiGLUCudaDescriptor_t desc);

#endif
