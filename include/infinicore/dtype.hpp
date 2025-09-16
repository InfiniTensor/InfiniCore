#ifndef __INFINICORE_DTYPE_API_HPP__
#define __INFINICORE_DTYPE_API_HPP__

#include <infinicore.h>

namespace infinicore {

enum class DataType {
    bfloat16 = INFINI_DTYPE_BF16,
    float16 = INFINI_DTYPE_F16,
    float32 = INFINI_DTYPE_F32,
    float64 = INFINI_DTYPE_F64,
    int32 = INFINI_DTYPE_I32,
    int64 = INFINI_DTYPE_I64,
    uint8 = INFINI_DTYPE_U8,
};

std::string to_string(const DataType &dtype);

} // namespace infinicore

#endif
