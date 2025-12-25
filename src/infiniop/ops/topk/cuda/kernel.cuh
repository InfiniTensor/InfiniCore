#ifndef __TOPK_CUDA_KERNEL_CUH__
#define __TOPK_CUDA_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>

namespace op::topk::cuda{
    __forceinline__ __device__ __host__ size_t baseOffsetExcludingDim(
        size_t flat_row,
        size_t ndim,
        const size_t *shape,
        const ptrdiff_t *strides,
        size_t dim) {
        size_t res = 0;
        for (size_t i = ndim; i-- > 0;) {
            if(i == dim) continue;
            res += (flat_row % shape[i]) * strides[i];
            flat_row /= shape[i];
        }
        return res;
    }

    __forceinline__ __device__ __host__ size_t indexToOffset(
        size_t flat_index,
        size_t ndim,
        const size_t *shape,
        const ptrdiff_t *strides) {
        size_t res = 0;
        for (size_t i = ndim; i-- > 0;) {
            res += (flat_index % shape[i]) * strides[i];
            flat_index /= shape[i];
        }
        return res;
    }

    template<typename Tdata>
    __device__ __forceinline__ float to_float(Tdata v);

    template<>
    __device__ __forceinline__ float to_float<float>(float v) { return v;}

    template<>
    __device__ __forceinline__ float to_float<half>(half v) { return __half2float(v);}
    
    template<>
    __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v);}

    #ifdef ENABLE_MOORE_API
    template<>
    __device__ __forceinline__ float to_float<__mt_bfloat16>(__mt_bfloat16 v) { return __bfloat162float(v);}
    #endif

    // float -> ordered uint32
    __device__ __forceinline__ uint32_t float_to_uint_ordered(float value) {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
        uint32_t mask = (uint32_t)(-((int32_t)bits >> 31)) | 0x80000000u;
        return bits ^ mask;
    }

    // -------------------------------------------
    // gather: input(strided) -> cur(row-major)
    // cur_idx initialized to [0..n-1]
    // -------------------------------------------
    template<typename Tdata>
    __global__ void gather_rowwise(const Tdata*  input, Tdata* cur_vals, size_t* cur_idx,
                                size_t rows, size_t n, size_t ndim, size_t dim, const size_t*  shape, const ptrdiff_t*  strides){
        size_t row = blockIdx.y;
        size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if(row >= rows || i >= n) return;
        size_t base = baseOffsetExcludingDim(row, ndim, shape, strides, dim);
        size_t off = base + i * strides[dim];

        cur_vals[row * n + i] = input[off];
        cur_idx[row * n + i] = i; // 索引初始化为[0..n-1]
    }
    // -------------------------------------------
    // per-row state
    // cur_n: current candidate count in cur (<= n)
    // rem_k: remaining k to fill
    // out_pos: already appended count into sel
    // -------------------------------------------
    __global__ void init_row_state(size_t* cur_n, size_t* rem_k, size_t* out_pos, size_t rows, size_t n, size_t k){
        size_t r = blockIdx.x * blockDim.x + threadIdx.x;
        if(r < rows){
            cur_n[r] = n;
            rem_k[r] = k;
            out_pos[r] = 0;
        }
    }

    __global__ void zero_row_counters(size_t* ones_count, size_t* zeros_count, size_t rows){
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if(r < rows){
            ones_count[r] = 0;
            zeros_count[r] = 0;
        }
    }

    // -------------------------------------------
    // radix partition (rowwise)
    // - partition cur into ones/zeros by bit_pos on key(value)
    // - largest: keep bit=1 side as "better"; smallest uses ~key so still keep bit=1
    // -------------------------------------------
    template<size_t BLOCK_SIZE, typename Tdata>
    __global__ void partition_rowwise(const Tdata* cur_vals, size_t* cur_idx, Tdata* ones_vals, size_t* ones_idx,
                                    Tdata* zeros_vals, size_t* zeros_idx, const size_t* cur_n, size_t rows, size_t n,
                                    size_t bit_pos, bool largest, size_t* ones_count, size_t* zeros_count){
        size_t row = blockIdx.y;
        if(row >= rows)return;

        __shared__ Tdata sh1_vals[BLOCK_SIZE];
        __shared__ size_t sh1_idx[BLOCK_SIZE];
        __shared__ Tdata sh0_vals[BLOCK_SIZE];
        __shared__ size_t sh0_idx[BLOCK_SIZE];
        __shared__ int sh1_n, sh0_n;
        __shared__ size_t base1, base0;

        size_t tid = threadIdx.x;
        if(tid == 0){sh1_n = 0; sh0_n = 0;}
        __syncthreads();

        size_t i = blockIdx.x * blockDim.x + tid;
        size_t cn = cur_n[row];
        if(i < cn){
            size_t off = row * n + i;
            Tdata v = cur_vals[off];
            size_t idx = cur_idx[off];

            size_t key = float_to_uint_ordered(to_float<Tdata>(v));
            if(!largest) key = ~key;
            size_t b = (key >> bit_pos) & 1;

            if(b){
                size_t p = atomicAdd(&sh1_n, 1);
                sh1_vals[p] = v;
                sh1_idx[p] = idx;
            } else {
                size_t p = atomicAdd(&sh0_n, 1);
                sh0_vals[p] = v;
                sh0_idx[p] = idx;
            }
        }
        __syncthreads();

        if(tid == 0){
            base1 = atomicAdd(&ones_count[row], sh1_n);
            base0 = atomicAdd(&zeros_count[row], sh0_n);
        }
        __syncthreads();

        for(size_t j = tid; j < sh1_n; j += blockDim.x){
            size_t o = row * n + base1 + j;
            ones_vals[o] = sh1_vals[j];
            ones_idx[o] = sh1_idx[j];
        }
        for(size_t j = tid; j < sh0_n; j += blockDim.x){
            size_t o = row * n + base0 + j;
            zeros_vals[o] = sh0_vals[j];
            zeros_idx[o] = sh0_idx[j];
        }
    }

    // -------------------------------------------
    // decide + append ones + compact next cur
    // -------------------------------------------
    template<size_t BLOCK_SIZE, typename Tdata>
    __global__ void decide_and_compact(Tdata* cur_vals, size_t* cur_idx, const Tdata* ones_vals, const size_t* ones_idx, const Tdata* zeros_vals, const size_t* zeros_idx,
                                        const size_t* ones_count, const size_t*  zeros_count, size_t* cur_n, size_t* rem_k, size_t* out_pos, 
                                        Tdata* sel_vals, size_t* sel_idx, size_t rows, size_t n, size_t k){
            size_t row = blockIdx.x;
            if(row >= rows)return;
            size_t tid = threadIdx.x;
            size_t rem = rem_k[row];
            if(rem <= 0)return;
            size_t oc = ones_count[row];
            size_t zc = zeros_count[row];
            size_t pos = out_pos[row];

            bool keep_ones = (oc >= rem);
            if(!keep_ones){
                for(size_t j = tid; j < oc; j += blockDim.x){
                    if(pos + j < k){
                        size_t o = row * n + j;
                        sel_vals[row * k + pos + j] = ones_vals[o];
                        sel_idx[row * k + pos + j] = ones_idx[o];
                    }
                }
            }
            __syncthreads();
            if(tid == 0){
                if(keep_ones){
                    cur_n[row] = oc;
                } else {
                    out_pos[row] = pos + oc;
                    rem_k[row] = rem - oc;
                    cur_n[row] = zc;
                }
            }
            __syncthreads();
            size_t new_n = cur_n[row];
            for(size_t j = tid; j < new_n; j += blockDim.x){
                size_t o = row * n + j;
                cur_vals[o] = keep_ones ? ones_vals[o] : zeros_vals[o];
                cur_idx[o] = keep_ones ? ones_idx[o] : zeros_idx[o];
            }
    }

    // -------------------------------------------
    // finalize: append remaining from cur
    // -------------------------------------------
    template<size_t BLOCK_SIZE, typename Tdata>
    __global__ void take_remaining(const Tdata* cur_vals, const size_t* cur_idx, const size_t* cur_n, const size_t* rem_k, const size_t* out_pos,
                                   Tdata* sel_vals, size_t* sel_idx, size_t rows, size_t n, size_t k){
            size_t row = blockIdx.x;
            size_t tid = threadIdx.x;
            if(row >= rows) return;
            size_t rem = rem_k[row];
            size_t pos = out_pos[row];
            size_t cn = cur_n[row];

            size_t take = rem;
            if(take > cn) take = cn;
            for(size_t j = tid; j < take; j+=blockDim.x){
                if(pos + j < k){
                    size_t o = row * k + pos + j;
                    sel_vals[o] = cur_vals[row * n + j];
                    sel_idx[o] = cur_idx[row * n + j];
                }
            }
    }
    // -------------------------------------------
    // sort: per-row CUB BlockRadixSort on sel[row, k]
    // - use (key=value_mapped) sort, and use "pos" to gather (value, idx)
    // - require k <= BLOCK_SIZE * ITEMS_PER_THREAD
    // -------------------------------------------
    template<size_t BLOCK_SIZE, size_t ITEMS_PER_THREAD, typename Tdata>
    __global__ void sort_sel_rowwise(const Tdata* sel_in_vals, const size_t* sel_in_idx, Tdata* sel_out_vals, size_t* sel_out_idx, size_t rows, size_t k, bool largest){
        size_t row = blockIdx.x;
        if(row >= rows) return;
        using BlockSort = cub::BlockRadixSort<uint32_t, BLOCK_SIZE, ITEMS_PER_THREAD, size_t>;
        __shared__ typename BlockSort::TempStorage temp_storage;
        uint32_t keys[ITEMS_PER_THREAD];
        size_t pos[ITEMS_PER_THREAD];
        #pragma unroll
        for (size_t it = 0; it < ITEMS_PER_THREAD; ++it) {
            size_t j = threadIdx.x + it * BLOCK_SIZE;
            if (j < k) {
                Tdata v = sel_in_vals[row * k + j];
                uint32_t key = float_to_uint_ordered(to_float<Tdata>(v));
                if (!largest) key = ~key;
                keys[it] = key;
                pos[it]  = j;
            } else {
                // invalid items: set minimal key, pos=0 (will be ignored by valid_items)
                keys[it] = 0u;
                pos[it]  = 0;
            }
        }

        // 使用 valid_items，让 padding 不参与排序结果
        const int valid_items = k;
        BlockSort(temp_storage).SortDescending(keys, pos, valid_items);

        __syncthreads();

        #pragma unroll
        for (size_t it = 0; it < ITEMS_PER_THREAD; ++it) {
            size_t out_j = threadIdx.x + it * BLOCK_SIZE;
            if (out_j < k) {
                size_t src_j = pos[it];
                sel_out_vals[row * k + out_j] = sel_in_vals[row * k + src_j];
                sel_out_idx[row * k + out_j]  = sel_in_idx[row * k + src_j];
            }
        }
    }

    // -------------------------------------------
    // scatter: sel[row,k] -> output(strided)
    // output is contiguous
    // -------------------------------------------
    template <typename Tdata>
    __global__ void scatter_to_output(const Tdata* sel_vals, const size_t*  sel_idx, Tdata*  values_out, size_t*  indices_out, 
                                    size_t rows, size_t k, size_t ndim, size_t dim, const size_t*  out_shape, const ptrdiff_t*  out_strides) {

            size_t row = blockIdx.y;
            size_t j   = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= rows || j >= k) return;

            size_t base = baseOffsetExcludingDim(row, ndim, out_shape, out_strides, dim);
            size_t off  = base + j * out_strides[dim];

            values_out[off]  = sel_vals[row * k + j];
            indices_out[off] = sel_idx[row * k + j];
    }


}// namespace op::topk::cuda
// 后续可结合bitonic sort在kernel中进行排序

#endif // __TOPK_CUDA_KERNEL_H__
