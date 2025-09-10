#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/conv1d.h"

#ifdef ENABLE_CPU_API
// No CPU implementation for MVP
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/conv1d_nvidia.cuh"
#endif

__C __export infiniStatus_t infiniopCreateConv1dDescriptor(
    infiniopHandle_t handle,
    infiniopConv1dDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    size_t kernel_size)
{
#define CREATE(CASE, NS) \
  case CASE: \
    return op::conv1d::NS::Descriptor::create( \
      handle, reinterpret_cast<op::conv1d::NS::Descriptor **>(desc_ptr), \
      y_desc, x_desc, w_desc, kernel_size)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
      CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
      CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
      return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C __export infiniStatus_t infiniopGetConv1dWorkspaceSize(
    infiniopConv1dDescriptor_t desc,
    size_t *size)
{
#define GET(CASE, NS) \
  case CASE: \
    *size = reinterpret_cast<const op::conv1d::NS::Descriptor*>(desc)->workspaceSize(); \
    return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
      GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
      GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
      return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C __export infiniStatus_t infiniopConv1dFn(
    infiniopConv1dDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w,
    void *stream)
{
#define CALL(CASE, NS) \
  case CASE: \
    return reinterpret_cast<const op::conv1d::NS::Descriptor*>(desc) \
      ->fn(workspace, workspace_size, y, x, w, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
      CALL(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
      CALL(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
      return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALL
}

__C __export infiniStatus_t infiniopConv1dUpdate(
    infiniopConv1dUpdateParams_t *params)
{
#define UPDATE(CASE, NS) \
  case CASE: \
    return reinterpret_cast<const op::conv1d::NS::Descriptor*>(params->desc)->update((void*)params)

    switch (params->desc->device_type) {
#ifdef ENABLE_NVIDIA_API
      UPDATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
      UPDATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
      return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef UPDATE
}

__C __export infiniStatus_t infiniopDestroyConv1dDescriptor(
    infiniopConv1dDescriptor_t desc)
{
#define DELETE(CASE, NS) \
  case CASE: \
    delete reinterpret_cast<const op::conv1d::NS::Descriptor*>(desc); \
    return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
      DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
      DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
      return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}
