#ifndef __INFINIUTILS_CUSTOM_TYPES_H__
#define __INFINIUTILS_CUSTOM_TYPES_H__
#include <stdint.h>
#include <type_traits>

struct CustomFloat16 {
    uint16_t _v;
};
typedef struct CustomFloat16 fp16_t;

struct CustomBFloat16 {
    uint16_t _v;
};
typedef struct CustomBFloat16 bf16_t;

bool _f16_to_bool(fp16_t val);
float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);

bool _bf16_to_bool(bf16_t val);
float _bf16_to_f32(bf16_t val);
bf16_t _f32_to_bf16(float val);

namespace utils {
// General template for non-fp16_t conversions
template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, bf16_t>::value) {
        return _f32_to_f16(_bf16_to_f32(val));
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, fp16_t>::value) {
        return _f32_to_bf16(_f16_to_f32(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, bool>::value) {
        return static_cast<TypeTo>(_bf16_to_bool(val));
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && std::is_same<TypeTo, bool>::value) {
        return static_cast<TypeTo>(_f16_to_bool(val));
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
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
        return _bf16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_bf16_to_f32(val));
    } else {
        // float tmp;
        // if constexpr (std::is_same<TypeFrom, fp16_t>::value){
        //     tmp = _f16_to_f32(val);
        // }
        // else if constexpr (std::is_same<TypeFrom, bf16_t>::value){
        //     tmp = _bf16_to_f32(val);
        // }
        // else{
        //     tmp = static_cast<float>(val);
        // }
        // if constexpr (std::is_same<TypeTo, fp16_t>::value){
        //     return _f32_to_f16(tmp);
        // }
        // else if constexpr (std::is_same<TypeFrom, bf16_t>::value){
        //     return _f32_to_bf16(tmp);
        // }
        // else{
        // return static_cast<TypeTo>(tmp);
        // }
        return static_cast<TypeTo>(val);
    }
}

} // namespace utils

#endif
