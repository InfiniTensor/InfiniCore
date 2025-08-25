#ifndef __LAYER_NORM_NVIDIA_CUH__
#define __LAYER_NORM_NVIDIA_CUH__

#include "../layer_norm.h"

namespace op::layer_norm::nvidia {

class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;
    LayerNormInfo info;
    size_t _workspace_size;

    Descriptor(Opaque *opaque, LayerNormInfo info, size_t workspace_size, infiniDevice_t device, int device_id)
        : InfiniopDescriptor{device, device_id}, _opaque(opaque), info(std::move(info)), _workspace_size(workspace_size) {}

public:
    ~Descriptor();

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

} // namespace op::layer_norm::nvidia

#endif // __LAYER_NORM_NVIDIA_CUH__