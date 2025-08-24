#ifndef __BATCH_NORM_BACKWARD_METAX_H__
#define __BATCH_NORM_BACKWARD_METAX_H__

#include "../batch_norm_backward.h"

namespace op::batch_norm_backward::metax {

struct BatchNormBackwardInfo {
    size_t _batch_size;
    size_t _channels;
    size_t _spatial_size;
    size_t total_elements;
    size_t input_size;
    size_t output_size;
    
    infiniDtype_t dtype;
    infiniDtype_t wtype;
    infiniDtype_t btype;
    infiniDtype_t atype;
    float momentum;
    float eps;
    bool has_bias;
    
    std::vector<size_t> grad_input_shape;
    std::vector<size_t> grad_weight_shape;
    std::vector<size_t> grad_bias_shape;
    std::vector<size_t> grad_output_shape;
    std::vector<size_t> input_shape;
    std::vector<size_t> weight_shape;
    std::vector<size_t> running_mean_shape;
    std::vector<size_t> running_var_shape;
    std::vector<size_t> shape;
    
    std::vector<ptrdiff_t> grad_input_strides;
    std::vector<ptrdiff_t> grad_weight_strides;
    std::vector<ptrdiff_t> grad_bias_strides;
    std::vector<ptrdiff_t> grad_output_strides;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> weight_strides;
    std::vector<ptrdiff_t> running_mean_strides;
    std::vector<ptrdiff_t> running_var_strides;
    
    size_t ndim() const { return shape.size(); }
    size_t channels() const { return _channels; }
    size_t batch_size() const { return _batch_size; }
    size_t spatial_size() const { return _spatial_size; }
};

class Descriptor final : public InfiniopDescriptor {
public:
    BatchNormBackwardInfo info;
    size_t workspace_size;

    Descriptor(BatchNormBackwardInfo info, size_t workspace_size, infiniDevice_t device, int device_id)
        : InfiniopDescriptor{device, device_id}, info(std::move(info)), workspace_size(workspace_size) {}

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

// 模板函数声明
template<typename T>
infiniStatus_t batchNormBackwardMetax(
    const BatchNormBackwardInfo &info,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    const void *running_mean,
    const void *running_var,
    void *workspace,
    void *stream);

} // namespace op::batch_norm_backward::metax

#endif // __BATCH_NORM_BACKWARD_METAX_H__