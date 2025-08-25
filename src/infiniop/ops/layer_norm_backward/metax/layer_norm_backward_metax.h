#ifndef __LAYER_NORM_BACKWARD_METAX_H__
#define __LAYER_NORM_BACKWARD_METAX_H__

#include "../layer_norm_backward.h"

namespace op::layer_norm_backward::metax {

class Descriptor final : public InfiniopDescriptor {
    LayerNormBackwardInfo _info;
    size_t _workspace_size;

    Descriptor(LayerNormBackwardInfo info, size_t workspace_size, infiniDevice_t device, int device_id)
        : InfiniopDescriptor{device, device_id}, _info(std::move(info)), _workspace_size(workspace_size) {}

public:
    ~Descriptor() = default;

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

// 模板函数声明
template<typename T>
infiniStatus_t layerNormBackwardMetax(
    const LayerNormBackwardInfo &info,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    const void *input_std_deviation,
    const void *input_standardization,
    void *stream);

} // namespace op::layer_norm_backward::metax

#endif // __LAYER_NORM_BACKWARD_METAX_H__