#ifndef __LAYER_NORM_BACKWARD_NVIDIA_CUH__
#define __LAYER_NORM_BACKWARD_NVIDIA_CUH__

#include "../layer_norm_backward.h"

namespace op::layer_norm_backward::nvidia {

class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;
    LayerNormBackwardInfo _info;
    size_t _workspace_size;

    Descriptor(Opaque *opaque, LayerNormBackwardInfo info, size_t workspace_size, infiniDevice_t device, int device_id)
        : InfiniopDescriptor{device, device_id}, _opaque(opaque), _info(std::move(info)), _workspace_size(workspace_size) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t grad_input_desc,
        infiniopTensorDescriptor_t grad_weight_desc,
        infiniopTensorDescriptor_t grad_bias_desc,
        infiniopTensorDescriptor_t grad_output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t input_std_deviation_desc,
        infiniopTensorDescriptor_t input_standardization_desc,
        float epsilon);

    infiniStatus_t get_workspace_size(size_t *size) const;

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *grad_input,
        void *grad_weight,
        void *grad_bias,
        const void *grad_output,
        const void *input,
        const void *weight,
        const void *input_std_deviation,
        const void *input_standardization,
        void *stream) const;
};

} // namespace op::layer_norm_backward::nvidia

#endif // __LAYER_NORM_BACKWARD_NVIDIA_CUH__