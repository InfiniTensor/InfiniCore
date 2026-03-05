#ifndef __BITWISE_RIGHT_SHIFT_CPU_H__
#define __BITWISE_RIGHT_SHIFT_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cstdint>
#include <type_traits>

ELEMENTWISE_DESCRIPTOR(bitwise_right_shift, cpu)

namespace op::bitwise_right_shift::cpu {
typedef struct BitwiseRightShiftOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &x, const T &shift) const {
        constexpr unsigned kBits = static_cast<unsigned>(sizeof(T) * 8);
        using TUnsigned = std::make_unsigned_t<T>;
        using WideUnsigned = std::conditional_t<(kBits <= 32), uint32_t, uint64_t>;
        using WideSigned = std::conditional_t<(kBits <= 32), int32_t, int64_t>;

        const WideUnsigned s = static_cast<WideUnsigned>(static_cast<TUnsigned>(shift)) & (kBits - 1u);

        if constexpr (std::is_signed_v<T>) {
            const WideSigned xw = static_cast<WideSigned>(x);
            return static_cast<T>(xw >> static_cast<unsigned>(s));
        } else {
            const WideUnsigned xw = static_cast<WideUnsigned>(static_cast<TUnsigned>(x));
            return static_cast<T>(xw >> static_cast<unsigned>(s));
        }
    }
} BitwiseRightShiftOp;
} // namespace op::bitwise_right_shift::cpu

#endif // __BITWISE_RIGHT_SHIFT_CPU_H__
