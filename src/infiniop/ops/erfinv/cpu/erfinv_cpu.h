// #ifndef __ERFINV_CPU_H__
// #define __ERFINV_CPU_H__

// #include "../../../elementwise/cpu/elementwise_cpu.h"
// #include <boost/math/special_functions/erf.hpp>

// ELEMENTWISE_DESCRIPTOR(erfinv, cpu)

// namespace op::erfinv::cpu {
// typedef struct ErfinvOp {
// public:
//     static constexpr size_t num_inputs = 1;
//     template <typename T>
//     T operator()(const T &x) const {
//         // Boost erf_inv typically takes float or double.
//         // We cast input to double for calculation and cast back to T.
//         // This handles float, double, fp16 (via implicit cast to
//         float/double), bf16 (via implicit cast). return
//         static_cast<T>(boost::math::erf_inv(static_cast<double>(x)));
//     }
// } ErfinvOp;
// } // namespace op::erfinv::cpu

// #endif // __ERFINV_CPU_H__

#ifndef __ERFINV_CPU_H__
#define __ERFINV_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(erfinv, cpu)

namespace op::erfinv::cpu {

template <typename T> static inline T erfinv_impl(T x) {
  double xd = static_cast<double>(x);

  if (xd <= -1.0)
    return -INFINITY;
  if (xd >= 1.0)
    return INFINITY;
  if (xd == 0.0)
    return 0.0;

  // Initial guess using Winitzki approximation
  double sign = xd < 0.0 ? -1.0 : 1.0;
  double abs_x = std::abs(xd);
  const double a = 0.147;
  double ln = std::log(1.0 - abs_x * abs_x);
  double t1 = 2.0 / (M_PI * a) + ln / 2.0;
  double t2 = ln / a;
  double y = std::sqrt(std::sqrt(t1 * t1 - t2) - t1);

  // Newton-Raphson iteration for higher precision
  // erfinv(x) = y, where erf(y) = x
  // f(y) = erf(y) - x = 0
  // f'(y) = 2/sqrt(pi) * exp(-y^2)
  const double sqrt_pi = 1.7724538509055160273;
  for (int i = 0; i < 3; ++i) {
    double erf_y = std::erf(y);
    double erf_prime = 2.0 / sqrt_pi * std::exp(-y * y);
    y = y - (erf_y - abs_x) / erf_prime;
  }

  return static_cast<T>(sign * y);
}

typedef struct ErfinvOp {
public:
  static constexpr size_t num_inputs = 1;

  template <typename T> T operator()(const T &x) const {
    return erfinv_impl(x);
  }
} ErfinvOp;

} // namespace op::erfinv::cpu

#endif // __ERFINV_CPU_H__
