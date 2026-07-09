#ifndef __INFINIOP_DSV4_MHC_PRE_H__
#define __INFINIOP_DSV4_MHC_PRE_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4MhcPreDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4MhcPreDescriptor(infiniopHandle_t handle,
                                                                      infiniopDsv4MhcPreDescriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t output_desc,
                                                                      infiniopTensorDescriptor_t input_desc,
                                                                      infiniopTensorDescriptor_t scale_desc,
                                                                      infiniopTensorDescriptor_t base_desc,
                                                                      float eps);

__INFINI_C __export infiniStatus_t infiniopGetDsv4MhcPreWorkspaceSize(infiniopDsv4MhcPreDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDsv4MhcPre(infiniopDsv4MhcPreDescriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *output,
                                                      const void *input,
                                                      const void *scale,
                                                      const void *base,
                                                      void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDsv4MhcPreDescriptor(infiniopDsv4MhcPreDescriptor_t desc);

#endif
