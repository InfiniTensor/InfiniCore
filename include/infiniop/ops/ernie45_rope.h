#ifndef __INFINIOP_ERNIE45_ROPE_API_H__
#define __INFINIOP_ERNIE45_ROPE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopErnie45MropeDescriptor_t;
typedef struct InfiniopDescriptor *infiniopErnie45VisionRopeDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateErnie45MropeDescriptor(
    infiniopHandle_t handle,
    infiniopErnie45MropeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t positions,
    double rope_theta,
    size_t section_h,
    size_t section_w,
    size_t section_t);

__INFINI_C __export infiniStatus_t infiniopGetErnie45MropeWorkspaceSize(infiniopErnie45MropeDescriptor_t desc,
                                                                        size_t *size);

__INFINI_C __export infiniStatus_t infiniopErnie45Mrope(
    infiniopErnie45MropeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *q,
    void *k,
    const void *positions,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyErnie45MropeDescriptor(infiniopErnie45MropeDescriptor_t desc);

__INFINI_C __export infiniStatus_t infiniopCreateErnie45VisionRopeDescriptor(
    infiniopHandle_t handle,
    infiniopErnie45VisionRopeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t positions,
    double rope_theta);

__INFINI_C __export infiniStatus_t infiniopGetErnie45VisionRopeWorkspaceSize(infiniopErnie45VisionRopeDescriptor_t desc,
                                                                             size_t *size);

__INFINI_C __export infiniStatus_t infiniopErnie45VisionRope(
    infiniopErnie45VisionRopeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *q,
    void *k,
    const void *positions,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyErnie45VisionRopeDescriptor(infiniopErnie45VisionRopeDescriptor_t desc);

#endif
