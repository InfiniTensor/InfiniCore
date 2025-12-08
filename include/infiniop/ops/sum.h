#ifndef __INFINIOP_SUM_API_H__
#define __INFINIOP_SUM_API_H__

#include "../operator_descriptor.h"
#include <cstddef>
#include <vector>
typedef struct InfiniopDescriptor *infiniopSumDescriptor_t;

__C __export infiniStatus_t infiniopCreateSumDescriptor(infiniopHandle_t handle,
                                                        infiniopSumDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t output_desc,
                                                        infiniopTensorDescriptor_t input_desc,
                                                        std::vector<size_t> dim, 
                                                        bool keepdim);

__C __export infiniStatus_t infiniopGetSumWorkspaceSize(infiniopSumDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSum(infiniopSumDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *output,
                                        const void *input,
                                        std::vector<size_t> dim, 
                                        bool keepdim,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroySumDescriptor(infiniopSumDescriptor_t desc);

#endif
