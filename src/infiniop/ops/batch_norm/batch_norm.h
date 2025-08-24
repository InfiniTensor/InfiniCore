#ifndef __BATCH_NORM_H__
#define __BATCH_NORM_H__

#include "../../operator.h"
#include "../../tensor.h"
#include "infinicore.h"

namespace op::batch_norm {

struct BatchNormInfo {
    size_t batch_size;
    size_t channels;
    size_t spatial_size;  // H * W for 3D input (N, C, H*W)
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

#define DESCRIPTOR(NAMESPACE)                                                                      \
    namespace NAMESPACE {                                                                          \
        struct Descriptor : public InfiniopDescriptor {                                           \
            BatchNormInfo info;                                                                    \
                                                                                                   \
            static infiniStatus_t create(                                                          \
                infiniopHandle_t handle,                                                           \
                Descriptor **desc_ptr,                                                             \
                infiniopTensorDescriptor_t output_desc,                                            \
                infiniopTensorDescriptor_t input_desc,                                             \
                infiniopTensorDescriptor_t weight_desc,                                            \
                infiniopTensorDescriptor_t bias_desc,                                              \
                infiniopTensorDescriptor_t running_mean_desc,                                      \
                infiniopTensorDescriptor_t running_var_desc,                                       \
                float momentum,                                                                    \
                float eps);                                                                        \
                                                                                                   \
            infiniStatus_t get_workspace_size(size_t *size) const;                                \
                                                                                                   \
            infiniStatus_t calculate(                                                              \
                void *workspace, size_t workspace_size,                                            \
                void *output,                                                                      \
                const void *input,                                                                 \
                const void *weight,                                                                \
                const void *bias,                                                                  \
                void *running_mean,                                                                \
                void *running_var,                                                                 \
                void *stream) const;                                                               \
        };                                                                                         \
    }

} // namespace op::batch_norm

#endif // __BATCH_NORM_H__