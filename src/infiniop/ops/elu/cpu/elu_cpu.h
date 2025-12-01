#ifndef __ELU_CPU_H__
#define __ELU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
// #include "../../../utils.h"

namespace op::elu::cpu {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::cpu::DeviceImpl> _device_info;
    size_t _workspace_size;
    float _alpha; // ELU parameter

    Descriptor(
        infiniDtype_t dtype,
        op::elementwise::ElementwiseInfo info,
        op::elementwise::cpu::DeviceImpl *device_info,
        size_t workspace_size,
        infiniDevice_t device_type,
        int device_id,
        float alpha)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)),
          _device_info(std::move(device_info)),
          _workspace_size(workspace_size),
          _alpha(alpha) {}

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec,
        float alpha);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};

typedef struct Eluop {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x, float alpha) const {
        if (x > T(0)) {
            return x;
        } else {
            return T(alpha) * (std::exp(x) - T(1));
        }
    }
} EluOp;
} // namespace op::elu::cpu

#endif // __ELU_CPU_H__