#ifndef __INFINIUTILS_CUSTOM_TYPES_H__
#define __INFINIUTILS_CUSTOM_TYPES_H__
#include <stdint.h>
#include <type_traits>
#include <cstring>

struct CustomFloat16 {
    uint16_t _v;
};
typedef struct CustomFloat16 fp16_t;

struct CustomBFloat16 {
    uint16_t _v;
};
typedef struct CustomBFloat16 bf16_t;

float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);

float _bf16_to_f32(bf16_t val);
bf16_t _f32_to_bf16(float val);

namespace utils {
// General template for non-fp16_t conversions
template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(val);
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && std::is_same<TypeTo, float>::value) {
        return _f16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_f16_to_f32(val));
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(val);
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, double>::value) {
        // 对于double到bf16的转换，先转换为float，但保留更高的精度
        float f_val = static_cast<float>(val);
        // 使用更高精度的舍入
        uint32_t bits32;
        std::memcpy(&bits32, &f_val, sizeof(bits32));
        
        // 截断前先加 0x7FFF，再根据第 16 位（有效位的最低位）的奇偶做 round-to-nearest-even
        const uint32_t rounding_bias = 0x00007FFF +          // 0111 1111 1111 1111
                                   ((bits32 >> 16) & 1); // 尾数的有效位的最低位奇数时 +1，即实现舍入偶数

        uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);
        
        return bf16_t{bf16_bits};
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value && !std::is_same<TypeFrom, double>::value) {
        return _f32_to_bf16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
        return _bf16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_bf16_to_f32(val));
    } else {
        return static_cast<TypeTo>(val);
    }
}

} // namespace utils

#endif
