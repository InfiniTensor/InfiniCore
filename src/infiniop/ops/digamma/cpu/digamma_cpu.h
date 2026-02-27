#ifndef __DIGAMMA_CPU_H__
#define __DIGAMMA_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(digamma, cpu)

namespace op::digamma::cpu {

// Digamma function implementation using asymptotic expansion
template <typename T>
T digamma_impl(T x) {
    // Handle special cases
    if (x <= 0.0) return std::numeric_limits<T>::quiet_NaN();
    
    // Use recurrence relation: digamma(x+1) = digamma(x) + 1/x
    // Reduce to x in [1, 2] range
    T result = 0.0;
    while (x < 1.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    while (x > 2.0) {
        x -= 1.0;
        result += 1.0 / x;
    }
    
    // For x in [1, 2], use series expansion
    // digamma(x) ≈ -gamma - 1/x + sum(k=1 to inf) x/(k*(k+x))
    // Simplified approximation for [1, 2]
    const T gamma = 0.57721566490153286060651209008240243104215933593992; // Euler-Mascheroni constant
    result -= gamma;
    result -= 1.0 / x;
    
    // Add series terms (truncated)
    T sum = 0.0;
    for (int k = 1; k <= 20; ++k) {
        sum += x / (static_cast<T>(k) * (static_cast<T>(k) + x));
    }
    result += sum;
    
    return result;
}

typedef struct DigammaOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return digamma_impl(x);
    }
} DigammaOp;
} // namespace op::digamma::cpu

#endif // __DIGAMMA_CPU_H__
