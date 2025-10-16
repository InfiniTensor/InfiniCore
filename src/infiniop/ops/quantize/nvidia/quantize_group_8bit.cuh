#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../../../devices/nvidia/nvidia_common.cuh"

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
    unsigned mask = 0xffff;

    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
    return val;
}

template <
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void per_token_group_quant_8bit_kernel(
    const T *__restrict__ input,
    void *__restrict__ output_q,
    scale_packed_t *__restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float min_8bit,
    const float max_8bit,
    const int num_groups_per_row = 0,
    const int scale_stride = 0) {
    const int threads_per_group = 16;
    const int64_t local_group_id = threadIdx.x / threads_per_group;
    const int lane_id = threadIdx.x % threads_per_group;

    const int64_t block_group_id = blockIdx.x * groups_per_block;
    const int64_t global_group_id = block_group_id + local_group_id;
    const int64_t block_group_offset = global_group_id * group_size;

    float local_absmax = eps;

    using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
    static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

    const T *group_input = input + block_group_offset;
    DST_DTYPE *group_output = static_cast<DST_DTYPE *>(output_q) + block_group_offset;
    scale_element_t *scale_output;

    if constexpr (IS_COLUMN_MAJOR) {
        const int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
        const int row_idx = global_group_id / num_groups_per_row;
        const int col_idx_unpacked = global_group_id % num_groups_per_row;
        const int col_idx = col_idx_unpacked / num_elems_per_pack;
        const int pack_idx = col_idx_unpacked % num_elems_per_pack;
        scale_output = reinterpret_cast<scale_element_t *>(output_s) + (col_idx * scale_stride * num_elems_per_pack + row_idx * num_elems_per_pack + pack_idx);
    } else {
        static_assert(!SCALE_UE8M0);
        scale_output = output_s + global_group_id;
    }

    constexpr uint32_t vec_size = 16 / sizeof(T);

    const int32_t num_vec_elems = group_size / vec_size;

    for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
        T *input_vec = new T[vec_size];
        for (int j = 0; j < vec_size; j++) {
            input_vec[j] = *(group_input + i * vec_size + j);
        }

#pragma unroll
        for (uint32_t j = 0; j < vec_size; ++j) {
            float val = static_cast<float>(input_vec[j]);
            float abs_val = fabsf(val);
            local_absmax = fmaxf(local_absmax, abs_val);
        }
        delete input_vec;
    }

    local_absmax = GroupReduceMax(local_absmax, lane_id);

    float y_s = local_absmax / max_8bit;
    if constexpr (SCALE_UE8M0) {
        y_s = exp2f(ceilf(log2f(fmaxf(y_s, 1e-10f))));
    }

    // TODO can optimize
    scale_element_t y_s_quant;
    if constexpr (SCALE_UE8M0) {
        y_s_quant = (uint8_t)(((int)log2f(y_s)) + 127);
    } else {
        y_s_quant = y_s;
    }

    if (lane_id == 0) {
        *scale_output = y_s_quant;
    }

    for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
        T *input_vec = new T[vec_size];
        for (int j = 0; j < vec_size; j++) {
            input_vec[j] = *(group_input + i * vec_size + j);
        }

#pragma unroll
        for (uint32_t j = 0; j < vec_size; ++j) {
            float val = static_cast<float>(input_vec[j]);
            float q_val = fminf(fmaxf(val / y_s, min_8bit), max_8bit);
            group_output[i * vec_size + j] = DST_DTYPE(q_val);
        }
        delete input_vec;
    }
}
