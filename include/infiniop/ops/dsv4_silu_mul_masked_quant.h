#ifndef __INFINIOP_DSV4_SILU_MUL_MASKED_QUANT_H__
#define __INFINIOP_DSV4_SILU_MUL_MASKED_QUANT_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4SiluMulMaskedQuantDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4SiluMulMaskedQuantDescriptor(infiniopHandle_t handle,
                                                                                  infiniopDsv4SiluMulMaskedQuantDescriptor_t *desc_ptr,
                                                                                  infiniopTensorDescriptor_t q_desc,
                                                                                  infiniopTensorDescriptor_t scale_desc,
                                                                                  infiniopTensorDescriptor_t gate_desc,
                                                                                  infiniopTensorDescriptor_t up_desc,
                                                                                  infiniopTensorDescriptor_t mask_desc);

__INFINI_C __export infiniStatus_t infiniopGetDsv4SiluMulMaskedQuantWorkspaceSize(infiniopDsv4SiluMulMaskedQuantDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4SiluMulMaskedQuant(infiniopDsv4SiluMulMaskedQuantDescriptor_t desc,
                                                                  void *workspace,
                                                                  size_t workspace_size,
                                                                  void *q,
                                                                  void *scale,
                                                                  const void *gate,
                                                                  const void *up,
                                                                  const void *mask,
                                                                  void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4SiluMulMaskedQuantDescriptor(infiniopDsv4SiluMulMaskedQuantDescriptor_t desc);

#endif
