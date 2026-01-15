#pragma once
#include <cuda_runtime.h>
#include <type_traits>

namespace op::cuda {

template <typename T>
__global__ void block_diag_kernel(
    T *output,
    const T **inputs,
    size_t num_inputs,
    size_t output_rows,
    size_t output_cols,
    size_t *row_offsets,
    size_t *col_offsets,
    size_t *input_rows,
    size_t *input_cols) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = output_rows * output_cols;

    if (idx >= total) return;

    size_t out_row = idx / output_cols;
    size_t out_col = idx % output_cols;

    // Find which input matrix this output position belongs to
    for (size_t i = 0; i < num_inputs; ++i) {
        size_t row_start = row_offsets[i];
        size_t row_end = row_start + input_rows[i];
        size_t col_start = col_offsets[i];
        size_t col_end = col_start + input_cols[i];

        if (out_row >= row_start && out_row < row_end &&
            out_col >= col_start && out_col < col_end) {
            // This position belongs to input i
            size_t in_row = out_row - row_start;
            size_t in_col = out_col - col_start;
            size_t in_idx = in_row * input_cols[i] + in_col;
            output[idx] = inputs[i][in_idx];
            return;
        }
    }
    // Outside all blocks: should be zero (already initialized)
}

} // namespace op::cuda
