#ifndef __LEAKY_RELU_CPU_H__
#define __LEAKY_RELU_CPU_H__

#include <cmath>
#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

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
        std::vector<infiniopTensorDescriptor_t> input_descs,
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
        // LeakyReLU: x if x > 0, else negative_slope * x
        return x > static_cast<T>(0) ? x : static_cast<T>(negative_slope) * x;
    }
    
    // 为bf16类型特化，使用double作为中间计算类型以提高精度
    bf16_t operator()(const bf16_t &x, float negative_slope) const {
        // 将bf16转换为double进行计算，然后再转回bf16
        double x_double = static_cast<double>(_bf16_to_f32(x));
        // LeakyReLU计算
        double result = x_double > 0.0 ? x_double : static_cast<double>(negative_slope) * x_double;
        // 使用utils::cast从double直接转换到bf16，保留更高精度
        return utils::cast<bf16_t>(result);
    }
} LeakyReLUOp;
} // namespace op::leaky_relu::cpu

#endif // __LEAKY_RELU_CPU_H__