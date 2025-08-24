#ifndef __BATCH_NORM_METAX_H__
#define __BATCH_NORM_METAX_H__

#include "../batch_norm.h"

namespace op::batch_norm::metax {

struct BatchNormInfo {
    size_t batch_size;
    size_t channels;
    size_t spatial_size;
    size_t input_size;
    size_t output_size;
    
    infiniDtype_t dtype;
    float momentum;
    float eps;
    
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> weight_strides;
    std::vector<ptrdiff_t> bias_strides;
    std::vector<ptrdiff_t> running_mean_strides;
    std::vector<ptrdiff_t> running_var_strides;
};

class Descriptor final : public InfiniopDescriptor {
public:
    BatchNormInfo info;
    size_t workspace_size;

    Descriptor(BatchNormInfo info, size_t workspace_size, infiniDevice_t device, int device_id)
        : InfiniopDescriptor{device, device_id}, info(std::move(info)), workspace_size(workspace_size) {}

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t bias_desc,
        infiniopTensorDescriptor_t running_mean_desc,
        infiniopTensorDescriptor_t running_var_desc,
        float momentum,
        float eps);

    infiniStatus_t get_workspace_size(size_t *size) const;

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *output,
        const void *input,
        const void *weight,
        const void *bias,
        void *running_mean,
        void *running_var,
        void *stream) const;
};

// 模板函数声明
template<typename T>
infiniStatus_t batchNormMetax(
    const BatchNormInfo &info,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *running_mean,
    void *running_var,
    void *workspace,
    void *stream);

} // namespace op::batch_norm::metax

#endif // __BATCH_NORM_METAX_H__