#ifndef __INFINIOP_DSV4_DEEPGEMM_TF32_HC_PERNORM_GEMM_H__
#define __INFINIOP_DSV4_DEEPGEMM_TF32_HC_PERNORM_GEMM_H__

#include <stdint.h>

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4DeepgemmTf32HcPernormGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4DeepgemmTf32HcPernormGemmDescriptor(infiniopHandle_t handle,
                                                                                         infiniopDsv4DeepgemmTf32HcPernormGemmDescriptor_t *desc_ptr,
                                                                                         infiniopTensorDescriptor_t a_desc,
                                                                                         infiniopTensorDescriptor_t b_desc,
                                                                                         infiniopTensorDescriptor_t d_desc,
                                                                                         infiniopTensorDescriptor_t sqr_sum_desc,
                                                                                         int64_t num_splits);

__INFINI_C __export infiniStatus_t infiniopGetDsv4DeepgemmTf32HcPernormGemmWorkspaceSize(infiniopDsv4DeepgemmTf32HcPernormGemmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4DeepgemmTf32HcPernormGemm(infiniopDsv4DeepgemmTf32HcPernormGemmDescriptor_t desc,
                                                                         void *workspace,
                                                                         size_t workspace_size,
                                                                         const void *a,
                                                                         const void *b,
                                                                         void *d,
                                                                         void *sqr_sum,
                                                                         void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4DeepgemmTf32HcPernormGemmDescriptor(infiniopDsv4DeepgemmTf32HcPernormGemmDescriptor_t desc);

#endif
