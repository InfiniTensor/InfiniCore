#ifndef __CAST_CUDA_H__
#define __CAST_CUDA_H__

#include "../../../../utils/custom_types.h"

namespace op::cast::cuda {

struct CastOp {
public:
    static constexpr size_t num_inputs = 1;
    
    // 模板化的类型转换操作符
    template <typename Tout, typename Tin>
    __device__ __forceinline__ Tout operator()(const Tin &input) const {
        // 使用utils::cast进行类型转换
        return utils::cast<Tout>(input);
    }
};

} // namespace op::cast::cuda

#endif // __CAST_CUDA_H__