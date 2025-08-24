// #ifndef __WHERE_CUDA_H__
// #define __WHERE_CUDA_H__

// namespace op::where::cuda {
// typedef struct WhereOp {
// public:
//     static constexpr size_t num_inputs = 3;
//     template <typename T>
//     __device__ __forceinline__ T operator()(const bool &cond,const T &a_val, const T &b_val) const {
//         return cond ? a_val : b_val;
//     }
// } WhereOp;
// } // namespace op::where::cuda

// #endif // __WHERE_CUDA_H__
#ifndef __WHERE_CUDA_H__
#define __WHERE_CUDA_H__

namespace op::where::cuda {
typedef struct WhereOp {
public:
    static constexpr size_t num_inputs = 3;
    
    // 原有的operator()函数
    template <typename T>
    __device__ __forceinline__ T operator()(const bool &cond, const T &a_val, const T &b_val) const {
        return cond ? a_val : b_val;
    }
    
    // 为Metax兼容性添加的模板operator()函数
    template <typename Tout, typename... Tin>
    __device__ __forceinline__ Tout operator()(const Tin&... args) const {
        static_assert(sizeof...(Tin) == 3, "WhereOp expects exactly 3 arguments");
        const Tout& a_val = std::get<0>(std::tie(args...));
        const Tout& b_val = std::get<1>(std::tie(args...));
        const bool& cond = std::get<2>(std::tie(args...));
        return cond ? a_val : b_val;

    }
} WhereOp;
} // namespace op::where::cuda

#endif // __WHERE_CUDA_H__
