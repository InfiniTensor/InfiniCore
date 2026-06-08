#ifndef __INFINIOP_GPTQ_MARLIN_REPACK_API_H__
#define __INFINIOP_GPTQ_MARLIN_REPACK_API_H__

#include "../operator_descriptor.h"
#include <cstdint>

typedef struct InfiniopDescriptor *infiniopGptqMarlinRepackDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGptqMarlinRepackDescriptor(infiniopHandle_t handle,
                                                                            infiniopGptqMarlinRepackDescriptor_t *desc_ptr,
                                                                            infiniopTensorDescriptor_t output_desc,
                                                                            infiniopTensorDescriptor_t input_desc,
                                                                            infiniopTensorDescriptor_t perm_desc,
                                                                            int64_t num_bits,
                                                                            bool is_a_8bit);

__INFINI_C __export infiniStatus_t infiniopGetGptqMarlinRepackWorkspaceSize(infiniopGptqMarlinRepackDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopGptqMarlinRepack(infiniopGptqMarlinRepackDescriptor_t desc,
                                                            void *workspace,
                                                            size_t workspace_size,
                                                            void *output,
                                                            const void *input,
                                                            const void *perm,
                                                            void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGptqMarlinRepackDescriptor(infiniopGptqMarlinRepackDescriptor_t desc);

#endif
