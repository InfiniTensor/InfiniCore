#include "topkrouter_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::topkrouter::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t correction_bias_desc,
    size_t N, size_t width, size_t topk) {

    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    float *values, int *indices, void *x,
    float *correction_bias,
    float routed_scaling_factor,
    void *stream) const {

    return INFINI_STATUS_NOT_IMPLEMENTED;
}
} // namespace op::topkrouter::cpu
