#ifndef __BITWISE_RIGHT_SHIFT_MOORE_KERNEL_H__
#define __BITWISE_RIGHT_SHIFT_MOORE_KERNEL_H__

#include <cuda_runtime.h>
#include <type_traits>

namespace op::bitwise_right_shift::moore {

typedef struct BitwiseRightShiftOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const T &shift) const {
        return x >> shift;
    }
} BitwiseRightShiftOp;

} // namespace op::bitwise_right_shift::moore

#endif // __BITWISE_RIGHT_SHIFT_MOORE_KERNEL_H__
