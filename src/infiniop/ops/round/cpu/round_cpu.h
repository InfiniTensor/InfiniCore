#ifndef __ROUND_CPU_H__
#define __ROUND_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

namespace op::round::cpu {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    size_t _workspace_size;
    int _decimals;

    Descriptor(infiniDtype_t dtype,
               op::elementwise::ElementwiseInfo info,
               size_t workspace_size,
               infiniDevice_t device_type,
               int device_id,
               int decimals);

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec,
        int decimals);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;

    int decimals() const { return _decimals; }
};

typedef struct RoundOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x, int decimals) const {
        if (decimals == 0) {
            return static_cast<T>(std::nearbyint(static_cast<float>(x)));
        }
        float scale = std::pow(10.0f, static_cast<float>(decimals));
        float val = static_cast<float>(x) * scale;
        val = std::nearbyint(val);
        return static_cast<T>(val / scale);
    }
} RoundOp;

} // namespace op::round::cpu

#endif // __ROUND_CPU_H__