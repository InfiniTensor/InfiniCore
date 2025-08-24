#ifndef __LAYER_NORM_METAX_H__
#define __LAYER_NORM_METAX_H__

#include "../layer_norm.h"

namespace op::layer_norm::metax {

class Descriptor final : public InfiniopDescriptor {
    LayerNormInfo info;
    size_t _workspace_size;

    Descriptor(LayerNormInfo info, size_t workspace_size, infiniDevice_t device, int device_id)
        : InfiniopDescriptor{device, device_id}, info(std::move(info)), _workspace_size(workspace_size) {}

public:
    ~Descriptor() = default;

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t bias_desc,
        infiniopTensorDescriptor_t input_std_deviation_desc,
        infiniopTensorDescriptor_t input_standardization_desc,
        float eps);

    infiniStatus_t get_workspace_size(size_t *size) const;

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *output,
        const void *input,
        const void *weight,
        const void *bias,
        void *input_std_deviation,
        void *input_standardization,
        void *stream) const;
};

// 模板函数声明
template<typename T>
infiniStatus_t layerNormMetax(
    const LayerNormInfo &info,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *input_std_deviation,
    void *input_standardization,
    void *stream);

} // namespace op::layer_norm::metax

#endif // __LAYER_NORM_METAX_H__