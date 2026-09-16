// SPDX-License-Identifier: MIT
// Adapted from ROCm/AITER for InfiniCore's Hygon backend.

#pragma once
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "numeric_types.h"
#include <vector>

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_buffer_load_dword_lds(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

    int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
    int offset_s = gvOffset_s << shfl_count;
    int offset_v = gvOffset_v << shfl_count;

    asm volatile("s_mov_b32 m0, %1 \n\t"
                 "buffer_load_dword %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
                 "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
                 :);
}

template <class DataType, const int shfl_count>
__forceinline__ __device__ void inline_buffer_load_ushort(DataType &v_data, int v_offset, vec<uint, 4> global_addr, int s_offset) {

    int v_offset_bytes = v_offset << shfl_count;
    int s_offset_bytes = s_offset << shfl_count;

    // asm volatile(
    //     "buffer_load_dword %0, %1, %2, %3 offen\n"
    //     // "s_waitcnt vmcnt(0)\n"
    //     // "s_barrier\n"
    //     : "=v"(v_data)
    //     : "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bytes)
    //     : "memory");
    asm volatile(
        "buffer_load_ushort %0, %1, %2, %3 offen\n"
        // "s_waitcnt vmcnt(0)\n"
        // "s_barrier\n"
        : "=v"(v_data)
        : "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bytes)
        : "memory");
}

template <class DataType>
__forceinline__ __device__ void inline_buffer_load_ubyte(DataType &v_data, int v_offset, vec<uint, 4> global_addr, int s_offset) {

    int v_offset_bytes = v_offset;
    int s_offset_bytes = s_offset;

    // asm volatile(
    //     "buffer_load_dword %0, %1, %2, %3 offen\n"
    //     // "s_waitcnt vmcnt(0)\n"
    //     // "s_barrier\n"
    //     : "=v"(v_data)
    //     : "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bytes)
    //     : "memory");
    asm volatile(
        "buffer_load_ubyte %0, %1, %2, %3 offen\n"
        // "s_waitcnt vmcnt(0)\n"
        // "s_barrier\n"
        : "=v"(v_data)
        : "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bytes)
        : "memory");
}

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_buffer_load_dword(DataType &v_data, int v_offset, vec<uint, 4> global_addr, int s_offset) {

    int v_offset_bytes = v_offset << shfl_count;
    int s_offset_bytes = s_offset << shfl_count;

    asm volatile(

        "buffer_load_dword %0, %1, %2, %3 offen\n"

        // "s_waitcnt vmcnt(0)\n"
        // "s_barrier\n"
        : "=v"(v_data)
        : "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bytes)
        : "memory");
}

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_buffer_load_dwordx2_lds(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

    int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
    int offset_s = gvOffset_s << shfl_count;
    int offset_v = gvOffset_v << shfl_count;

    asm volatile("s_mov_b32 m0, %1 \n\t"
                 "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
                 "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
                 :);
}

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_buffer_load_dwordx4_lds(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

    int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
    int offset_s = gvOffset_s << shfl_count;
    int offset_v = gvOffset_v << shfl_count;

    asm volatile("s_mov_b32 m0, %1 \n\t"
                 "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
                 "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
                 :);
}

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_buffer_load_dwordx4(DataType &v_data, const vec4_uint global_addr, const int &gvOffset_s, const int &gvOffset_v) {

    int offset_s = gvOffset_s << shfl_count;
    int offset_v = gvOffset_v << shfl_count;

    asm volatile("buffer_load_dwordx4 %0, %1, %2 ,%3 offen  offset:0 \n"
                 : "=v"(v_data)
                 : "v"(offset_v), "s"(global_addr), "s"(offset_s)
                 : "memory");
}

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_s_load_dwordx4(DataType &s_data, const uint64_t global_addr, int gvOffset_s) {

    int offset_s = gvOffset_s << shfl_count;

    asm volatile("s_load_dwordx4 %0, %1 \n"
                 : "=s"(s_data)
                 : "s"(global_addr)
                 : "memory");
}

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_buffer_load_dword_lds_bypass_glc_slc(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

    int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
    int offset_s = gvOffset_s << shfl_count;
    int offset_v = gvOffset_v << shfl_count;

    asm volatile("s_mov_b32 m0, %1 \n\t"
                 "buffer_load_dword %0, %2, %3 ,offen  offset:0 glc slc lds\n" ::"v"(offset_v),
                 "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
                 :);
}

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_buffer_load_dword_lds_bypass_l1_glc(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

    int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
    int offset_s = gvOffset_s << shfl_count;
    int offset_v = gvOffset_v << shfl_count;

    asm volatile("s_mov_b32 m0, %1 \n\t"
                 "buffer_load_dword %0, %2, %3 ,offen  offset:0 glc lds\n" ::"v"(offset_v),
                 "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
                 :);
}

template <class DataType, const int shfl_count = 2>
__forceinline__ __device__ void inline_buffer_load_dword_lds_bypass_l2_slc(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

    int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
    int offset_s = gvOffset_s << shfl_count;
    int offset_v = gvOffset_v << shfl_count;

    asm volatile("s_mov_b32 m0, %1 \n\t"
                 "buffer_load_dword %0, %2, %3 ,offen  offset:0 slc lds\n" ::"v"(offset_v),
                 "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
                 :);
}

template <typename src_type = half_t, typename dst_type = float, const int dword_count = 1, const int auxilariy = 0>
__forceinline__ __device__ void builtin_buffer_load_dword_lds(src_type *const shared_addr, const vec4_uint rsrc, const int &lds_offset, const int gvOffset_s, const int &gvOffset_v) {
    constexpr int bytes_per_element = sizeof(dst_type);
    dst_type *ptr = reinterpret_cast<dst_type *>(shared_addr) + lds_offset;
    __builtin_hygon_raw_buffer_load_lds(
        rsrc,
        ptr,
        dword_count * 4,
        gvOffset_v * bytes_per_element,
        gvOffset_s * bytes_per_element,
        0,        /* immediate offset, instruction offset */
        auxilariy /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
    );
    // int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << 2);
    // int offset_s = gvOffset_s << 2;
    // int offset_v = gvOffset_v << 2;

    // asm volatile("s_mov_b32 m0, %1 \n\t"
    //             "buffer_load_dword %0, %2, %3 ,offen  offset:0, lds \n"
    //             :: "v"(offset_v), "s"(ldsAddrPerWave), "s"(rsrc), "s"(offset_s)
    //             :);
}

template <typename src_type = half_t, typename dst_type = float>
__forceinline__ __device__ void builtin_buffer_load_dword_lds_bypass_glc_slc(src_type *const shared_addr, const vec4_uint rsrc, const int &lds_offset, const int gvOffset_s, const int &gvOffset_v) {
    constexpr int bytes_per_element = sizeof(dst_type);
    dst_type *ptr = reinterpret_cast<dst_type *>(shared_addr) + lds_offset;
    __builtin_hygon_raw_buffer_load_lds(
        rsrc,
        ptr,
        4,
        gvOffset_v * bytes_per_element,
        gvOffset_s * bytes_per_element,
        0, /* immediate offset, instruction offset */
        11 /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
    );
}

template <class DataType, const int shfl_count>
__forceinline__ __device__ void inline_buffer_store_dword(const DataType v_data, const int &v_offset, const vec4_uint global_addr, const int &s_offset, const int s_constant = 0) {

    int v_offset_bytes = v_offset << shfl_count;
    int s_offset_bybes = s_offset << shfl_count;
    const int s_constant_bytes = s_constant << shfl_count;

    asm volatile(
        "buffer_store_dword %0, %1, %2, %3, offen  offset:%4 \n" ::"v"(v_data), "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bybes), "B"(s_constant_bytes)
        :);
}

template <class DataType, const int shfl_count>
__forceinline__ __device__ void inline_buffer_store_dword_glc_slc(DataType v_data, int &v_offset, vec4_uint global_addr, int &s_offset, const int s_constant = 0) {

    int v_offset_bytes = v_offset << shfl_count;
    int s_offset_bybes = s_offset << shfl_count;
    const int s_constant_bytes = s_constant << shfl_count;

    asm volatile(
        "buffer_store_dword %0, %1, %2, %3, offen  offset:%4  glc slc\n" ::"v"(v_data), "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bybes), "B"(s_constant_bytes)
        :);
}

template <typename VEC>
__forceinline__ __device__ void inline_ds_read_b32_no_wait(VEC *const shared_addr, const int &lds_offset, VEC &reg_val) {
    int ldsAddr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
    asm volatile(
        "ds_read_b32 %0, %1\n"
        : "=v"(reg_val)
        : "v"(ldsAddr)
        :);
}

template <typename VEC>
__forceinline__ __device__ void inline_ds_read_b32_no_wait_bytes(const int &lds_offset, VEC &reg_val) {
    asm volatile(
        "ds_read_b32 %0, %1\n"
        : "=v"(reg_val)
        : "v"(lds_offset)
        :);
}

template <typename VEC, typename dwordx2>
__forceinline__ __device__ void inline_ds_read2_b32_no_wait(VEC *shared_addr, const int &lds_offset, dwordx2 &reg_val, const int offset1) {
    int ldsAddr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
    asm volatile(
        "ds_read2_b32 %0 ,%1 offset0:0 offset1:%2\n"
        : "=v"(reg_val)
        : "v"(ldsAddr), "B"(offset1)
        :);
}

template <typename dwordx2>
__forceinline__ __device__ void inline_ds_read2_b32_no_wait_bytes(const int &lds_offset, dwordx2 &reg_val, const int offset1) {
    asm volatile(
        "ds_read2_b32 %0, %1 offset0:0 offset1:%2\n"
        : "=v"(reg_val)
        : "v"(lds_offset), "B"(offset1)
        :);
}

template <typename dwordx2>
__forceinline__ __device__ void inlineasm_fa_ds_read2_b32(float *shared_addr, const int &lds_offset, dwordx2 &reg_val, const int offset0, const int offset1) {
    int lds_addr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
    asm volatile(
        "ds_read2_b32 %0, %1 offset0:%2 offset1:%3\n"
        : "=v"(reg_val)
        : "v"(lds_addr), "B"(offset0), "B"(offset1)
        :);
}

__forceinline__ __device__ void inline_ds_write2_b32_no_wait_bytes(float *shared_addr, int lds_offset, const __float2 &reg_val, const int offset0, const int offset1) {
    int lds_addr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
    asm volatile(
        "ds_write2_b32 %0, %1, %2 offset0:%3 offset1:%4\n"
        : "=v"(lds_addr)
        : "v"(reg_val[0]), "v"(reg_val[1]), "B"(offset0), "B"(offset1)
        :);
}

template <typename VEC>
__forceinline__ __device__ void inline_ds_read_b32_wait(VEC *const shared_addr, const int &lds_offset, VEC &reg_val) {
    reg_val = shared_addr[lds_offset];
}

template <typename VEC>
__forceinline__ __device__ void inline_vgpr_init_zero(VEC &dst, const int idx) {
    asm("v_mov_b32 %0, 0x0"
        : "=v"(dst[idx])
        :);
}

template <typename VEC>
__forceinline__ __device__ void inline_vgpr4_init_zero(VEC &dst) {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    asm("v_mov_b64 %0, 0x0\n\t"
        "v_mov_b64 %1, 0x0\n\t"
        : "=v"(dst.u64[0]), "=v"(dst.u64[1])
        :);
#else
    asm("v_mov_b32 %0, 0x0\n\t"
        "v_mov_b32 %1, 0x0\n\t"
        "v_mov_b32 %2, 0x0\n\t"
        "v_mov_b32 %3, 0x0\n\t"
        : "=v"(dst.f32[0]), "=v"(dst.f32[1]), "=v"(dst.f32[2]), "=v"(dst.f32[3])
        :);
#endif
}

template <typename VEC>
__forceinline__ __device__ void inline_vgpr4_init_zero_4x4x4(VEC s_reg[4][4]) {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    asm("v_mov_b64 %0, 0x0\n\t"
        "v_mov_b64 %1, 0x0\n\t"
        "v_mov_b64 %2, 0x0\n\t"
        "v_mov_b64 %3, 0x0\n\t"
        "v_mov_b64 %4, 0x0\n\t"
        "v_mov_b64 %5, 0x0\n\t"
        "v_mov_b64 %6, 0x0\n\t"
        "v_mov_b64 %7, 0x0\n\t"
        "v_mov_b64 %8, 0x0\n\t"
        "v_mov_b64 %9, 0x0\n\t"
        "v_mov_b64 %10, 0x0\n\t"
        "v_mov_b64 %11, 0x0\n\t"
        "v_mov_b64 %12, 0x0\n\t"
        "v_mov_b64 %13, 0x0\n\t"
        "v_mov_b64 %14, 0x0\n\t"
        "v_mov_b64 %15, 0x0\n\t"
        "v_mov_b64 %16, 0x0\n\t"
        "v_mov_b64 %17, 0x0\n\t"
        "v_mov_b64 %18, 0x0\n\t"
        "v_mov_b64 %19, 0x0\n\t"
        "v_mov_b64 %20, 0x0\n\t"
        "v_mov_b64 %21, 0x0\n\t"
        "v_mov_b64 %22, 0x0\n\t"
        "v_mov_b64 %23, 0x0\n\t"
        "v_mov_b64 %24, 0x0\n\t"
        "v_mov_b64 %25, 0x0\n\t"
        "v_mov_b64 %26, 0x0\n\t"
        "v_mov_b64 %27, 0x0\n\t"
        "v_mov_b64 %28, 0x0\n\t"
        "v_mov_b64 %29, 0x0\n\t"
        "v_mov_b64 %30, 0x0\n\t"
        "v_mov_b64 %31, 0x0\n"
        : "=v"(s_reg[0][0].u64[0]), "=v"(s_reg[0][0].u64[1]), "=v"(s_reg[0][1].u64[0]), "=v"(s_reg[0][1].u64[1]), "=v"(s_reg[0][2].u64[0]), "=v"(s_reg[0][2].u64[1]), "=v"(s_reg[0][3].u64[0]), "=v"(s_reg[0][3].u64[1]), "=v"(s_reg[1][0].u64[0]), "=v"(s_reg[1][0].u64[1]), "=v"(s_reg[1][1].u64[0]), "=v"(s_reg[1][1].u64[1]), "=v"(s_reg[1][2].u64[0]), "=v"(s_reg[1][2].u64[1]), "=v"(s_reg[1][3].u64[0]), "=v"(s_reg[1][3].u64[1]), "=v"(s_reg[2][0].u64[0]), "=v"(s_reg[2][0].u64[1]), "=v"(s_reg[2][1].u64[0]), "=v"(s_reg[2][1].u64[1]), "=v"(s_reg[2][2].u64[0]), "=v"(s_reg[2][2].u64[1]), "=v"(s_reg[2][3].u64[0]), "=v"(s_reg[2][3].u64[1]), "=v"(s_reg[3][0].u64[0]), "=v"(s_reg[3][0].u64[1]), "=v"(s_reg[3][1].u64[0]), "=v"(s_reg[3][1].u64[1]), "=v"(s_reg[3][2].u64[0]), "=v"(s_reg[3][2].u64[1]), "=v"(s_reg[3][3].u64[0]), "=v"(s_reg[3][3].u64[1])
        :);
#else
    uint64_t pk_zero = 0x0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s_reg[i][j].u64[0] = pk_zero;
            s_reg[i][j].u64[1] = pk_zero;
        }
    }
#endif
}

template <typename VEC>
__forceinline__ __device__ void inline_vgpr4_init_zero_4x2x4(VEC s_reg[4][4]) {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    asm("v_mov_b64 %0, 0x0\n\t"
        "v_mov_b64 %1, 0x0\n\t"
        "v_mov_b64 %2, 0x0\n\t"
        "v_mov_b64 %3, 0x0\n\t"
        "v_mov_b64 %4, 0x0\n\t"
        "v_mov_b64 %5, 0x0\n\t"
        "v_mov_b64 %6, 0x0\n\t"
        "v_mov_b64 %7, 0x0\n\t"
        "v_mov_b64 %8, 0x0\n\t"
        "v_mov_b64 %9, 0x0\n\t"
        "v_mov_b64 %10, 0x0\n\t"
        "v_mov_b64 %11, 0x0\n\t"
        "v_mov_b64 %12, 0x0\n\t"
        "v_mov_b64 %13, 0x0\n\t"
        "v_mov_b64 %14, 0x0\n\t"
        "v_mov_b64 %15, 0x0\n\t"
        : "=v"(s_reg[0][0].u64[0]), "=v"(s_reg[0][0].u64[1]), "=v"(s_reg[0][2].u64[0]), "=v"(s_reg[0][2].u64[1]), "=v"(s_reg[1][0].u64[0]), "=v"(s_reg[1][0].u64[1]), "=v"(s_reg[1][2].u64[0]), "=v"(s_reg[1][2].u64[1]), "=v"(s_reg[2][0].u64[0]), "=v"(s_reg[2][0].u64[1]), "=v"(s_reg[2][2].u64[0]), "=v"(s_reg[2][2].u64[1]), "=v"(s_reg[3][0].u64[0]), "=v"(s_reg[3][0].u64[1]), "=v"(s_reg[3][2].u64[0]), "=v"(s_reg[3][2].u64[1])
        :);
#else
    uint64_t pk_zero = 0x0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
            s_reg[i][j].u64[0] = pk_zero;
            s_reg[i][j].u64[1] = pk_zero;
        }
    }
#endif
}

template <typename VEC>
__forceinline__ __device__ void inline_vgpr4_init_zero_1x4x4(VEC s_reg[1][4]) {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    asm("v_mov_b64 %0, 0x0\n\t"
        "v_mov_b64 %1, 0x0\n\t"
        "v_mov_b64 %2, 0x0\n\t"
        "v_mov_b64 %3, 0x0\n\t"
        "v_mov_b64 %4, 0x0\n\t"
        "v_mov_b64 %5, 0x0\n\t"
        "v_mov_b64 %6, 0x0\n\t"
        "v_mov_b64 %7, 0x0\n\t"
        : "=v"(s_reg[0][0].u64[0]), "=v"(s_reg[0][0].u64[1]), "=v"(s_reg[0][1].u64[0]), "=v"(s_reg[0][1].u64[1]), "=v"(s_reg[0][2].u64[0]), "=v"(s_reg[0][2].u64[1]), "=v"(s_reg[0][3].u64[0]), "=v"(s_reg[0][3].u64[1])
        :);
#endif
}

template <typename VEC>
__forceinline__ __device__ void inline_vgpr4_init_zero_1x2x4(VEC s_reg[1][4]) {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    asm("v_mov_b64 %0, 0x0\n\t"
        "v_mov_b64 %1, 0x0\n\t"
        "v_mov_b64 %2, 0x0\n\t"
        "v_mov_b64 %3, 0x0\n\t"
        : "=v"(s_reg[0][0].u64[0]), "=v"(s_reg[0][0].u64[1]), "=v"(s_reg[0][2].u64[0]), "=v"(s_reg[0][2].u64[1])
        :);
#endif
}

// d = a * b + c
inline __device__ __float2 inlineasm_fa_v_pk_fma_f32(__float2 &a, const __float2 &b, const __float2 &c) {
    __float2 d;
    asm volatile("v_pk_fma_f32 %0, %1, %2, %3  ; inlineasm_fa_v_pk_fma_f32"
                 : "=v"(d)
                 : "v"(a), "v"(b), "v"(c)
                 :);
    return d;
}

inline __device__ void inlineasm_fa_v_mov_b64(__float2 &c, const __float2 &a) {
    asm volatile("v_mov_b64 %0, %1 ; inlineasm_fa_v_mov_b64"
                 : "=v"(c)
                 : "v"(a)
                 :);
}

extern __device__ __attribute__((const)) __float2 __llvm_v_pk_fma_f32(__float2, __float2, __float2) __asm("llvm.fma.v2f32");

inline __device__ void inlineasm_fa_v_pk_mul_f32(__float2 &dst, const __float2 &src, const __float2 &scale) {
    asm volatile("v_pk_mul_f32 %0, %1, %2 ; inlineasm_fa_v_pk_mul_f32"
                 : "=v"(dst)
                 : "v"(src), "v"(scale)
                 :);
}

// c = a + b
inline __device__ void inlineasm_fa_v_pk_add_f32(__float2 &c, const __float2 &a, const __float2 &b) {
    asm volatile("v_pk_add_f32 %0, %1, %2 ; inlineasm_fa_v_pk_add_f32"
                 : "=v"(c)
                 : "v"(a), "v"(b)
                 :);
}

/*
    原来的 exp2f 对于极小数有特殊处理, 对于小于 -126 的输入 x , exp2f 计算方式是 2^(x + 64) * 2^{-64}
    但是对于深度学习来说, 2^-126 的数字其实没那么重要了, 因此只需要保留 v_exp_f32 直接暴力计算即可
*/
extern __device__ __attribute__((const)) float __llvm_exp2_f32(float) __asm("llvm.exp2.f32");
extern __device__ __attribute__((const)) float __llvm_log2_f32(float) __asm("llvm.log2.f32");
extern __device__ __attribute__((const)) float __llvm_fma_f32(float, float, float) __asm("llvm.fma.f32");
extern __device__ __attribute__((const)) int64_t __builtin_hygon_mov_b64(int64_t) __asm("llvm.hygon.mov64");

/* 直接内联汇编单独测试没问题, 但放到 flash attention 里面结果不对, 很奇怪 */
inline __device__ float inlineasm_fa_v_exp2_f32(const float x) {
    // return exp2f(x);
    float y;
    asm volatile(
        // "s_waitcnt lgkmcnt(0)\n\t"
        "v_exp_f32 %0, %1"
        : "=v"(y)
        : "v"(x)
        :);
    return y;
}

template <const int stride, typename T>
__device__ __forceinline__ vec<uint, 4> tcp_cache_swizzle_func(const T *ptr) {
    vec<uint, 4> res;
    *(uint64_t *)&res = reinterpret_cast<uint64_t>(ptr);
    if constexpr (stride == 256) {
        res[1] += 0x42000000; // 62 bit: cache swizzle;  48~61: Stride
    } else if constexpr (stride == 196) {
        res[1] += 0x41800000; // 62 bit: cache swizzle;  48~61: Stride
    } else if constexpr (stride == 128) {
        res[1] += 0x41000000; // stride 256 Bytes and change tagram
    } else if constexpr (stride == 64) {
        res[1] += 0x40800000; // stride 128 Bytes and change tagram
    }
    res[2] = 0x80000000;
    res[3] = 0x00020000;
    return res;
}

template <const int stride, typename T>
__device__ __forceinline__ vec<uint, 4> tcp_cache_swizzle_func_no(const T *ptr) {
    vec<uint, 4> res;
    *(uint64_t *)&res = reinterpret_cast<uint64_t>(ptr);
    // if constexpr (stride == 256) {
    //   res[1] += 0x42000000; // 62 bit: cache swizzle;  48~61: Stride
    // }
    // else if constexpr (stride == 196) {
    //     res[1] += 0x41800000; // 62 bit: cache swizzle;  48~61: Stride
    // } else if constexpr (stride == 128) {
    //     res[1] += 0x41000000; // stride 256 Bytes and change tagram
    // } else if constexpr (stride == 64) {
    //     res[1] += 0x40800000; // stride 128 Bytes and change tagram
    // }
    res[2] = 0x80000000;
    res[3] = 0x00020000;

    return res;
}

template <const int stride, typename T>
__device__ __forceinline__ vec<uint, 4> tcp_cache_swizzle_func_b8(const T *ptr) {
    vec<uint, 4> res;
    *(uint64_t *)&res = reinterpret_cast<uint64_t>(ptr);
    // if constexpr (stride == 256) {
    //   res[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    // }
    // else if constexpr (stride == 196) {
    //     res[1] += 0x40C00000; // 62 bit: cache swizzle;  48~61: Stride
    // } else if constexpr (stride == 128) {
    //     res[1] += 0x40800000; // stride 128 Bytes and change tagram
    // } else if constexpr (stride == 64) {
    //     res[1] += 0x40400000; // stride 64 Bytes and change tagram
    // }
    res[1] += 0x40800000;
    res[2] = 0x80000000;
    res[3] = 0x00020000;
    return res;
}

/*********************************************FP8 requirements************************************************ */

#define Check(X)                                \
    {                                           \
        if (X != hipSuccess) {                  \
            std::cout << "failed" << std::endl; \
        }                                       \
    }

#define MATRIX_LOAD_64x32_B8_LDS_REARRANGE(LDSADDR, SRSRC, MATRIX_OFFSET, R, T)                 \
    asm volatile("s_nop 0\n\t"                                                                  \
                 "matrix_load_64x16_b8 %0, %1, moffset:%2 " #R #T " lds\n\t"                    \
                 "matrix_load_64x16_b8 %0, %3, moffset:%4 " #R #T " lds\n" ::"s"(SRSRC),        \
                 "s"(LDSADDR), "n"(MATRIX_OFFSET), "s"(LDSADDR + 1024), "n"(MATRIX_OFFSET + 16) \
                 :);

#define DS_READ_MATRIX_64x16_B8(OFFSET, REG)                                                \
    int lds_offset_s = __builtin_amdgcn_readfirstlane(OFFSET);                              \
    asm volatile(                                                                           \
        "s_add_u32 m0, %1, 0x80000000\n\t"                                                  \
        "s_nop 0\n\t"                                                                       \
        "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 row:0x3 col:0x1 alt:0x0\n" \
        : "=v"(REG)                                                                         \
        : "s"(lds_offset_s)                                                                 \
        :);

#define MATRIX_LOAD_64x16_B8_LDS_TRANSPOSE(LDSADDR, SRSRC, MATRIX_OFFSET, R, T)            \
    LDSADDR = LDSADDR + 0x80000000;                                                        \
    asm volatile("s_nop 0\n\t"                                                             \
                 "matrix_load_64x16_b8 %0, %1, moffset:%2 " #R #T " lds\n\t" ::"s"(SRSRC), \
                 "s"(LDSADDR), "n"(MATRIX_OFFSET)                                          \
                 :);

template <int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_64x16_b8_lds_transpose(DataType *shared_addr, vec4_uint srsrc, int lds_offset, const int matrix_offset) {
#if defined(__gfx928__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    if constexpr (r && t) {
        MATRIX_LOAD_64x16_B8_LDS_TRANSPOSE(lds_addr_per_wave, srsrc, matrix_offset, r, t);

    } else if constexpr (r && !t) {
        MATRIX_LOAD_64x16_B8_LDS_TRANSPOSE(lds_addr_per_wave, srsrc, matrix_offset, r, );

    } else if constexpr (!r && t) {
        MATRIX_LOAD_64x16_B8_LDS_TRANSPOSE(lds_addr_per_wave, srsrc, matrix_offset, , t);

    } else {
        MATRIX_LOAD_64x16_B8_LDS_TRANSPOSE(lds_addr_per_wave, srsrc, matrix_offset, , );
    }
#endif
}

template <int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_64x32_b8_lds_rearrange(DataType *shared_addr, vec4_uint srsrc, int lds_offset, const int matrix_offset) {
#if defined(__gfx928__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    if constexpr (r && t) {
        MATRIX_LOAD_64x32_B8_LDS_REARRANGE(lds_addr_per_wave, srsrc, matrix_offset, r, t);

    } else if constexpr (r && !t) {
        MATRIX_LOAD_64x32_B8_LDS_REARRANGE(lds_addr_per_wave, srsrc, matrix_offset, r, );

    } else if constexpr (!r && t) {
        MATRIX_LOAD_64x32_B8_LDS_REARRANGE(lds_addr_per_wave, srsrc, matrix_offset, , t);

    } else {
        MATRIX_LOAD_64x32_B8_LDS_REARRANGE(lds_addr_per_wave, srsrc, matrix_offset, , );
    }
#endif
}

template <const int kINC, typename T, bool Do_CacheSwizzle = true>
__device__ __forceinline__ vec4_uint prepare_for_buffer_load(T *ptr, int stride) {
    vec4_uint res;
    *(uint64_t *)&res = reinterpret_cast<uint64_t>(ptr);
    // if constexpr (Do_CacheSwizzle) {
    //     if constexpr (kINC == 128) {
    //         res[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    //     } else if constexpr (kINC == 64) {
    //         res[1] += 0x40800000; // stride 128Bytes and change tagram
    //     }
    // }
    res[2] = stride;
    res[3] = 0x00000;
    return res;
}
