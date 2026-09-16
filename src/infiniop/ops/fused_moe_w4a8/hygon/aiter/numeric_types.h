// SPDX-License-Identifier: MIT
// Adapted from ROCm/AITER for InfiniCore's Hygon backend.

#pragma once
#include "hip/hip_bf16.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
using bhalf_t = __hip_bfloat16;
using half_t = __half;
using BFloat16 = bhalf_t;
using Float16 = half_t;
using Int32 = int;
using Int16 = unsigned short;
using Float32 = float;
using f8_t = uint8_t;

// fp8_e4m3 definitions
#ifndef AITER_FLOAT8_E4M3_T_DEFINED
#define AITER_FLOAT8_E4M3_T_DEFINED
struct alignas(1) Float8_e4m3_t {
    /// Data container
    uint8_t data;
    __host__ __device__ Float8_e4m3_t() = default;
    __host__ __device__ Float8_e4m3_t(uint8_t value) : data(value) {}
};
#endif
using vec_fp16 = __attribute__((__vector_size__(1 * sizeof(_Float16)))) _Float16;
using vec4_fp16 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using vec8_fp16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using vec2_fp16 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using vec16_fp16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;

using vec_bf16 = __attribute__((__vector_size__(1 * sizeof(unsigned short)))) unsigned short;
using vec4_bf16 = __attribute__((__vector_size__(4 * sizeof(unsigned short)))) unsigned short;
using vec8_bf16 = __attribute__((__vector_size__(8 * sizeof(unsigned short)))) unsigned short;
using vec2_bf16 = __attribute__((__vector_size__(2 * sizeof(unsigned short)))) unsigned short;
using vec16_bf16 = __attribute__((__vector_size__(16 * sizeof(unsigned short)))) unsigned short;

using vec4_uint = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using vec2_uint = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;

using vec_fp8 = __attribute__((__vector_size__(1 * sizeof(uint8_t)))) uint8_t;
using vec4_fp8 = __attribute__((__vector_size__(4 * sizeof(uint8_t)))) uint8_t;
using vec8_fp8 = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;
using vec2_fp8 = __attribute__((__vector_size__(2 * sizeof(uint8_t)))) uint8_t;
using vec16_fp8 = __attribute__((__vector_size__(16 * sizeof(uint8_t)))) uint8_t;

using vec_int8 = __attribute__((__vector_size__(1 * sizeof(int8_t)))) int8_t;
using vec4_int8 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;
using vec8_int8 = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using vec2_int8 = __attribute__((__vector_size__(2 * sizeof(int8_t)))) int8_t;
using vec16_int8 = __attribute__((__vector_size__(16 * sizeof(int8_t)))) int8_t;

using vec4_int32 = __attribute__((__vector_size__(4 * sizeof(int)))) int;

using __builtin_half2 = __attribute__((ext_vector_type(2))) __fp16;
using __float2 = __attribute__((ext_vector_type(2))) float;

using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;

union union_vec8_int8 {
    int8_t i8[8];
    int i32[2];
};

union union_vec16_fp8 {
    int8_t i8[16];
    vec8_int8 i8x8[2];
    vec4_fp32 f32x4;
    vec4_int32 i32x4;
};

union union_vec4_fp16 {
    vec4_fp16 fp16;
    half2 input_half2[2];
};

union union_vec4_i32 {
    vec4_fp32 fp32;
    float4 input_float4;
    vec4_uint vec4_i32;
    int i32[4];
};

template <typename FloatType>
union reg_bf16_fp16;

template <>
union reg_bf16_fp16<half2> {
    vec2_fp16 vec2_i16[4];
    __half2 bf162[4];
};

template <>
union reg_bf16_fp16<__hip_bfloat162> {
    vec2_bf16 vec2_i16[4];
    __hip_bfloat162 bf162[4];
};

template <typename FloatType>
union union_vec8_fp16;

template <>
union union_vec8_fp16<half> {
    vec8_fp16 fp16;
    half2 input_half2[4];
};

template <>
union union_vec8_fp16<__hip_bfloat16> {
    Int16 fp16[8];
    __hip_bfloat162 input_half2[4];
};

union old_union_vec4_i32 {
    vec4_fp32 fp32;
    float4 input_float4;
    vec4_uint i32;
};

union union_i32 {
    float fp32;
    uint i32;
};

union union_vec4_fp32 {
    vec4_fp32 f32;
    double data[2];
    __float2 u64[2];
    vec2_fp32 v64[2];
};

union union_vec2_fp32 {
    vec2_fp32 f32;
    double data;
    __float2 u64;
};

union union_vec4_uint {
    unsigned long long u64[2]; // 128 bits
    uint4 u32;
    uint8_t u8[16];
};

union union_vec4_int8 {
    int i32;
    int8_t i8[4];
};

template <int len>
union union_vec_array {
    int i32[len];
    vec4_int32 vec4_i32[len / 4];
};

union union_vec2_uint {
    uint2 u32;
    unsigned long long u64;
};

template <typename T>
using vec_Element = std::conditional_t<
    std::is_same_v<T, half_t>,
    T,
    std::conditional_t<
        std::is_same_v<T, bhalf_t>,
        Int16,
        void>>;

template <typename T>
using vec2_Element = std::conditional_t<
    std::is_same_v<T, half_t>,
    vec2_fp16,
    std::conditional_t<
        std::is_same_v<T, bhalf_t>,
        vec2_bf16,
        std::conditional_t<
            std::is_same_v<T, Float8_e4m3_t> || std::is_same_v<T, int8_t>,
            vec2_fp8,
            std::conditional_t<
                std::is_same_v<T, int8_t>,
                vec2_int8,
                void>>>>;

template <typename T>
using vec4_Element = std::conditional_t<
    std::is_same_v<T, half_t>,
    vec4_fp16,
    std::conditional_t<
        std::is_same_v<T, bhalf_t>,
        vec4_bf16,
        std::conditional_t<
            std::is_same_v<T, Float8_e4m3_t>,
            vec4_fp8,
            std::conditional_t<
                std::is_same_v<T, int8_t>,
                vec4_int8,
                void>>>>;

template <typename T>
using vec8_Element = std::conditional_t<
    std::is_same_v<T, half_t>,
    vec8_fp16,
    std::conditional_t<
        std::is_same_v<T, bhalf_t>,
        vec8_bf16,
        std::conditional_t<
            std::is_same_v<T, Float8_e4m3_t>,
            vec8_fp8,
            std::conditional_t<
                std::is_same_v<T, int8_t>,
                vec8_int8,
                void>>>>;

template <typename Element>
union union_vec2_f16x2 {
    vec2_fp32 f32;
    double data;
    __float2 u64;
    vec2_Element<Element> f16x2[2];
    vec4_Element<Element> f8x4[2];
    Element f16[4];
    Element f8[8];
};

template <typename Element>
union union_vec4_f16x2 {
    vec4_fp32 f32;
    double data[2];
    __float2 u64[2];
    vec2_Element<Element> f16x4[4];
    vec4_Element<Element> f8x4[4];
};

template <typename T>
using vec4_Accum = std::conditional_t<std::is_same_v<T, float>, union_vec4_fp32, vec4_bf16>;

template <typename T>
using vec2_Accum = std::conditional_t<std::is_same_v<T, float>, union_vec2_fp32, vec4_bf16>;

template <typename T>
__forceinline__ __device__ vec4_Element<T> make_vec4_f16(T a, T b, T c, T d) {
    return {a, b, c, d};
}

template <>
__forceinline__ __device__ vec4_Element<bhalf_t> make_vec4_f16(bhalf_t a, bhalf_t b, bhalf_t c, bhalf_t d) {
#ifdef ROCM_5_7
    return {a.data, b.data, c.data, d.data};
#else
    // return {__hip_bfloat16_raw(a).x, __hip_bfloat16_raw(b).x, __hip_bfloat16_raw(c).x, __hip_bfloat16_raw(d).x};
    return {*(unsigned short *)(&a), *(unsigned short *)(&b), *(unsigned short *)(&c), *(unsigned short *)(&d)};
#endif
}

template <class T, class AccumType>
inline __device__ vec4_fp32 mmac(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3) {
#if defined(__gfx916__) || defined(__gfx926__)
    return {0, 0, 0, 0};
#else
    return __builtin_hcu_mmac_f32_16x16x16_bf16(v1, v2, v3);
#endif
}

template <>
inline __device__ vec4_fp32 mmac<half_t, float>(const vec4_fp16 &v1, const vec4_fp16 &v2, const vec4_fp32 &v3) {
#if defined(__gfx916__) || defined(__gfx926__)
    return {0, 0, 0, 0};
#else
    return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#endif
}

template <>
inline __device__ vec4_fp32 mmac<__hip_bfloat16, float>(const vec4_bf16 &v1, const vec4_bf16 &v2, const vec4_fp32 &v3) {
#if defined(__gfx916__) || defined(__gfx926__)
    return {0, 0, 0, 0};
#else
    return __builtin_hcu_mmac_f32_16x16x16_bf16(v1, v2, v3);
#endif
}

/* ****************************************************************optimize kernel requirements */

using half2_t = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using half4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using half8_t = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using v4bh = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using uintx4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int8x16 = __attribute__((__vector_size__(16 * sizeof(int8_t)))) char;

template <typename scalar_t, size_t len>
using vec = __attribute__((__vector_size__(len * sizeof(scalar_t)))) scalar_t;

template <typename scalar_t>
union vec_element_8 {};

template <>
union vec_element_8<bhalf_t> {
    vec8_bf16 data;
};

template <>
union vec_element_8<half_t> {
    vec8_fp16 data;
};

template <typename scalar_t>
union vec_element_2 {};

template <>
union vec_element_2<bhalf_t> {
    vec2_bf16 data;
};

template <>
union vec_element_2<half_t> {
    vec2_fp16 data;
};

// fp8_e4m3 definitions
//  struct alignas(1) Float8_e4m3_t{
//      /// Data container
//      uint8_t data;
//      __host__ __device__ Float8_e4m3_t() = default;
//      __host__ __device__ Float8_e4m3_t(uint8_t value): data(value) {}
//  };

// template <typename Element, size_t len>
// struct vec {
//     using type = __attribute__((__vector_size__(len * sizeof(Element)))) Element;
// };

// // 特化：为 __half 类型提供专门的实现
// template <size_t len>
// struct vec<__half, len> {
//     using type = __attribute__((__vector_size__(len * sizeof(uint16_t)))) _Float16;
// };

// // 特化：为 BFloat16 类型提供专门的实现
// template <size_t len>
// struct vec<BFloat16, len> {
//     using type = __attribute__((__vector_size__(len * sizeof(uint16_t)))) unsigned short;
// };

// // 特化：为 BFloat16 类型提供专门的实现
// template <size_t len>
// struct vec<int8_t, len> {
//     using type = __attribute__((__vector_size__(len * sizeof(int8_t)))) char;
// };

// // 特化：为 int 类型提供专门的实现
// template <size_t len>
// struct vec<int, len> {
//     using type = __attribute__((__vector_size__(len * sizeof(int)))) int;
// };

// // 特化：为 float 类型提供专门的实现
// template <size_t len>
// struct vec<float, len> {
//     using type = __attribute__((__vector_size__(len * sizeof(float)))) float;
// };

// template <size_t len>
// struct vec<uint32_t, len> {
//     using type = __attribute__((__vector_size__(len * sizeof(uint32_t)))) uint32_t;
// };

// template <typename Element, size_t len>
// union union_vec{
//   int8_t int8_array[len * sizeof(Element)];
//   Element scalar_array[len];
//   typename vec<Element, 2>::type scalar2_array[len/2];
//   int int_array[len * sizeof(Element) / 4];
//   float float_array[len * sizeof(Element) / 4];
//   int32_t uint_array[len * sizeof(Element) / 4];
//   int64_t uint64_array[len * sizeof(Element) / 8];
//   vec<int8_t, 8>::type int8t_array[len * sizeof(Element) / 8];
//   vec<int, 2>::type int2_array[len * sizeof(Element) / 8];
//   vec<int, 4>::type int4_array[len * sizeof(Element) / 16];
//   vec<float, 4>::type float4_array[len * sizeof(Element) / 16];
// };
