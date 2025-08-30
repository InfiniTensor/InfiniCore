#ifndef LEAKY_RELU_CPU_H
#define LEAKY_RELU_CPU_H

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

namespace op::leaky_relu::cpu {
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::cpu::DeviceImpl> _device_info;
    size_t _workspace_size;
    float _negative_slope;

    Descriptor(
        infiniDtype_t dtype,
        op::elementwise::ElementwiseInfo info,
        op::elementwise::cpu::DeviceImpl *device_info,
        size_t workspace_size,
        infiniDevice_t device_type,
        int device_id,
        float negative_slope)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)),
          _device_info(std::move(device_info)),
          _workspace_size(workspace_size),
          _negative_slope(negative_slope) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc,
        float negative_slope);

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};

typedef struct LeakyReLUOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x, float negative_slope) const {
        // LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
        // Equivalent to: x >= 0 ? x : negative_slope * x
        if constexpr (std::is_same_v<T, fp16_t>) {
            float x_f = static_cast<float>(x);
            float result = x_f >= 0.0f ? x_f : negative_slope * x_f;
            return static_cast<T>(result);
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            float x_f = static_cast<float>(x);
            float result = x_f >= 0.0f ? x_f : negative_slope * x_f;
            return static_cast<T>(result);
        } else if constexpr (std::is_same_v<T, float>) {
            return x >= 0.0f ? x : negative_slope * x;
        } else if constexpr (std::is_same_v<T, double>) {
            return x >= 0.0 ? x : static_cast<double>(negative_slope) * x;
        } else {
            return x >= T(0) ? x : static_cast<T>(negative_slope) * x;
        }
    }

} LeakyReLUOp;

} // namespace op::leaky_relu::cpu

#endif // LEAKY_RELU_CPU_H