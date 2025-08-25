#ifndef __BATCH_NORM_BACKWARD_CPU_H__
#define __BATCH_NORM_BACKWARD_CPU_H__

#include "../batch_norm_backward.h"

namespace op::batch_norm_backward::cpu {

class Descriptor final : public InfiniopDescriptor {
public:
    BatchNormBackwardInfo info;

    Descriptor(infiniDevice_t device, int device_id, BatchNormBackwardInfo info)
        : InfiniopDescriptor{device, device_id}, info(std::move(info)) {}

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t grad_input_desc,
        infiniopTensorDescriptor_t grad_weight_desc,
        infiniopTensorDescriptor_t grad_bias_desc,
        infiniopTensorDescriptor_t grad_output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t running_mean_desc,
        infiniopTensorDescriptor_t running_var_desc,
        float momentum,
        float eps);

    infiniStatus_t get_workspace_size(size_t *size) const;

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *grad_input,
        void *grad_weight,
        void *grad_bias,
        const void *grad_output,
        const void *input,
        const void *weight,
        const void *running_mean,
        const void *running_var,
        void *stream) const;
};

} // namespace op::batch_norm_backward::cpu

#endif // __BATCH_NORM_BACKWARD_CPU_H__