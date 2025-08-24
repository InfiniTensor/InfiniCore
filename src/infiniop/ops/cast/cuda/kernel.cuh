#ifndef __CAST_CUDA_H__
#define __CAST_CUDA_H__

template <typename Src, typename Dst>
__device__ __forceinline__ Dst convert_cast(const Src& x) {
    // return utils::cast<Dst>(x);
    return static_cast<Dst>(x);
}

template <>
__device__ __forceinline__ float convert_cast<half, float>(const half& x) {
    return __half2float(x);
}

// 特化2：uint64_t → __half（新增，解决歧义转换问题）
// 显式将 uint64_t 转为 unsigned long long（匹配 __half 的明确构造函数）
template <>
__device__ __forceinline__ __half convert_cast<uint64_t, __half>(const uint64_t& x) {
    // 步骤1：先转 unsigned long long（匹配 __half(const unsigned long long val) 构造函数）
    // 步骤2：再转 __half，消除编译器歧义
    return static_cast<__half>(static_cast<unsigned long long>(x));
}

namespace op::cast::cuda {
typedef struct CastOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T, typename U>
    __device__ __forceinline__ T operator()(const U &a) const {
        return convert_cast<U, T>(a);
    }

    // 用于 elementwise 内核的显式模板调度（Elementwise Kernel 会显式传 <Tout, Tin...>）
    template <typename Tout, typename... Tin>
    __device__ __forceinline__ Tout operator()(const Tin&... args) const {
        static_assert(sizeof...(Tin) == 1, "CastOp expects exactly 1 input");
        const auto &x = std::get<0>(std::tie(args...));
        return (*this).operator()<decltype(x), Tout>(x);
    }
} CastOp;
} // namespace op::cast::cuda

#endif // __CAST_CUDA_H__
