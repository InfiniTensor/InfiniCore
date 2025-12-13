#ifndef __FMIN_CUDA_H__
#define __FMIN_CUDA_H__

namespace op::fmin::cuda {
typedef struct FminOp {
public:
  static constexpr size_t num_inputs = 2;
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    if constexpr (std::is_same_v<T, half2>) {
      return __hmin2(a, b);
    } else if constexpr (std::is_same_v<T, half> ||
                         std::is_same_v<T, cuda_bfloat16>) {
      return __hmin(a, b);
    } else if constexpr (std::is_same_v<T, float>) {
      return fminf(a, b);
    } else {
      return a < b ? a : b;
    }
  }
} FminOp;
} // namespace op::fmin::cuda

#endif // __ADD_CUDA_H__
