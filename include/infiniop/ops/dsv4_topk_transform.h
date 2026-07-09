#ifndef __INFINIOP_DSV4_TOPK_TRANSFORM_API_H__
#define __INFINIOP_DSV4_TOPK_TRANSFORM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDsv4TopkTransformDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDsv4TopkTransformDescriptor(infiniopHandle_t handle, infiniopDsv4TopkTransformDescriptor_t *desc_ptr, infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_tables_desc, int page_size);
__INFINI_C __export infiniStatus_t infiniopGetDsv4TopkTransformWorkspaceSize(infiniopDsv4TopkTransformDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopDsv4TopkTransform(infiniopDsv4TopkTransformDescriptor_t desc, void *workspace, size_t workspace_size, void *out, const void *scores, const void *seq_lens, const void *page_tables, void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyDsv4TopkTransformDescriptor(infiniopDsv4TopkTransformDescriptor_t desc);

#endif // __INFINIOP_DSV4_TOPK_TRANSFORM_API_H__
