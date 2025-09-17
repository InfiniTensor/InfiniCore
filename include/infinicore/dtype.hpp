#ifndef __INFINICORE_DTYPE_API_HPP__
#define __INFINICORE_DTYPE_API_HPP__

#include <infinicore.h>

namespace infinicore {

enum class DataType {
    BFLOAT16 = INFINI_DTYPE_BF16,
    FLOAT16 = INFINI_DTYPE_F16,
    FLOAT32 = INFINI_DTYPE_F32,
    FLOAT64 = INFINI_DTYPE_F64,
    INT32 = INFINI_DTYPE_I32,
    INT64 = INFINI_DTYPE_I64,
    UINT8 = INFINI_DTYPE_U8,
};

std::string toString(const DataType &dtype);

} // namespace infinicore

#endif
