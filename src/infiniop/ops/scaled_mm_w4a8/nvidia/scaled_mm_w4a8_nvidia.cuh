#ifndef __SCALED_MM_W4A8_NVIDIA_H__
#define __SCALED_MM_W4A8_NVIDIA_H__

#include "../scaled_mm_w4a8.h"

DESCRIPTOR(nvidia)
namespace op::scaled_mm_w4a8::nvidia {
infiniStatus_t prepareGlmW4A16Awq(
    infiniopTensorDescriptor_t qweight_desc,
    infiniopTensorDescriptor_t qzeros_desc,
    infiniopTensorDescriptor_t scales_desc,
    infiniopTensorDescriptor_t checkpoint_weight_desc,
    infiniopTensorDescriptor_t channel_scales_desc,
    void *qweight, void *qzeros, void *scales,
    const void *checkpoint_weight, const void *channel_scales, void *stream);
}

#endif
