#ifndef __DIV_CPU_H__
#define __DIV_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

// ELEMENTWISE_DESCRIPTOR(div, cpu)

namespace op::div::cpu {
class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::cpu::DeviceImpl> _device_info;
    size_t _workspace_size;

    Descriptor(
        infiniDtype_t dtype,
        op::elementwise::ElementwiseInfo info,
        op::elementwise::cpu::DeviceImpl *device_info,
        size_t workspace_size,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)),
          _device_info(std::move(device_info)),
          _workspace_size(workspace_size) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs);

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        int mode,
        void *stream) const;
};

typedef struct DivOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a / b;
    }
} DivOp;
typedef struct DivOpTrunc {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, fp16_t>) {
            float af = _f16_to_f32(a);
            float bf = _f16_to_f32(b);
            float res = std::trunc(af / bf);

            return _f32_to_f16(float(res));
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            float af = _bf16_to_f32(a);
            float bf = _bf16_to_f32(b);
            float res = std::trunc(af / bf);
            return _f32_to_bf16(res);
        } else if constexpr (std::is_same_v<T, float>) {
            float res = std::trunc(a / b);
            return res;
        } else if constexpr (std::is_same_v<T, double>) {
            double res = std::trunc(a / b);
            return res;
        } else {
            return std::trunc(a / b);
        }
    }
} DivOpTrunc;
typedef struct DivOpFloor {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, fp16_t>) {
            float af = _f16_to_f32(a);
            float bf = _f16_to_f32(b);
            float res = std::floor(af / bf);

            return _f32_to_f16(float(res));
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            float af = _bf16_to_f32(a);
            float bf = _bf16_to_f32(b);
            float res = std::floor(af / bf);

            return _f32_to_bf16(res);
        } else if constexpr (std::is_same_v<T, float>) {
            float res = std::floor(a / b);
            return res;
        } else if constexpr (std::is_same_v<T, double>) {
            double res = std::floor(a / b);
            return res;
        } else {
            return std::floor(a / b);
        }
    }
} DivOpFloor;
} // namespace op::div::cpu

#endif // __DIV_CPU_H__
