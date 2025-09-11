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
    infiniopTensorDescriptor_t b_desc,
    void *pads,
    void *strides,
    void *dilations,
    size_t n)
{
#define CREATE(CASE, NS) \
  case CASE: \
    return op::conv1d::NS::Descriptor::create( \
      handle, reinterpret_cast<op::conv1d::NS::Descriptor **>(desc_ptr), \
      y_desc, x_desc, w_desc, b_desc, pads, strides, dilations, n)

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

/**
 * @brief Implementation of the 1D convolution operator.
 *
 * This function dispatches the 1D convolution operation to the appropriate
 * backend implementation based on the device type specified in the descriptor.
 */
extern "C" __export infiniStatus_t infiniopConv1d(
    infiniopConv1dDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w, const void *bias,
    void *stream)
{
#define CALCULATE(CASE, NS) \
  case CASE: \
    return reinterpret_cast<const op::conv1d::NS::Descriptor*>(desc) \
      ->calculate(workspace, workspace_size, y, x, w, bias, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
      CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
      CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
    default:
      return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
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
