#ifndef __INFINIOP_TOPK_API_H__
#define __INFINIOP_TOPK_API_H__

#include "../operator_descriptor.h"
#include <cstddef>
#include <vector>
typedef struct InfiniopDescriptor *infiniopTopKDescriptor_t;
// # Test cases format: (shape, input_strides, k, dim, largest, sorted)
// Returns the k largest elements of the given input tensor along a given dimension.
// If dim is not given, the last dimension of the input is chosen.
// If largest is False then the k smallest elements are returned.
// A namedtuple of (values, indices) is returned with the values and indices of the largest k elements of each row of the input tensor in the given dimension dim.
// The boolean option sorted if True, will make sure that the returned k elements are themselves sorted
__C __export infiniStatus_t infiniopCreateTopKDescriptor(infiniopHandle_t handle,
                                                        infiniopTopKDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t values_output_desc,
                                                        infiniopTensorDescriptor_t indices_output_desc,
                                                        infiniopTensorDescriptor_t input_desc,
                                                        size_t k,
                                                        size_t dim,
                                                        bool largest,
                                                        bool sorted);

__C __export infiniStatus_t infiniopGetTopKWorkspaceSize(infiniopTopKDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTopK(infiniopTopKDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *values_output,
                                        void *indices_output,
                                        const void *input,
                                        size_t k,
                                        size_t dim,
                                        bool largest,
                                        bool sorted,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyTopKDescriptor(infiniopTopKDescriptor_t desc);

#endif
