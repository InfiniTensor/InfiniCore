#ifndef __INFINIOP_ROPE_V2_CUDA_KERNEL_CUH__
#define __INFINIOP_ROPE_V2_CUDA_KERNEL_CUH__

template <typename Tdata, typename Tindex, typename Tangle>
__device__ void ropeThreadPerItemBlock(
    Tdata *y_,
    const Tdata *x_,
    const Tindex *__restrict__ pos_ids,
    const Tangle *__restrict__ sin_table,
    const Tangle *__restrict__ cos_table,
    size_t table_dim,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead) {

    auto y_offset = blockIdx.x * y_stride_seqlen + blockIdx.y * y_stride_nhead;
    auto x_offset = blockIdx.x * x_stride_seqlen + blockIdx.y * x_stride_nhead;
    size_t pos_id = size_t(pos_ids[blockIdx.x]);
    auto table_offset = pos_id * table_dim;
    const size_t half_dim = table_dim; // Head dimension = 2 * table_dim

    for (size_t i = threadIdx.x; i < table_dim; i += blockDim.x) {
        Tangle sin__ = sin_table[table_offset + i];
        Tangle cos__ = cos_table[table_offset + i];

        // Calculate positions in first and second halves
        size_t pos0 = i;
        size_t pos1 = i + half_dim;

        if constexpr (std::is_same<Tdata, half>::value) {
            Tangle x0 = __half2float(x_[x_offset + pos0]);
            Tangle x1 = __half2float(x_[x_offset + pos1]);

            Tangle y0 = x0 * cos__ - x1 * sin__;
            Tangle y1 = x0 * sin__ + x1 * cos__;

            y_[y_offset + pos0] = __float2half(y0);
            y_[y_offset + pos1] = __float2half(y1);
        } else if constexpr (std::is_same<Tdata, cuda_bfloat16>::value) {
            Tangle x0 = __bfloat162float(x_[x_offset + pos0]);
            Tangle x1 = __bfloat162float(x_[x_offset + pos1]);

            Tangle y0 = x0 * cos__ - x1 * sin__;
            Tangle y1 = x0 * sin__ + x1 * cos__;

            y_[y_offset + pos0] = __float2bfloat16(y0);
            y_[y_offset + pos1] = __float2bfloat16(y1);
        } else {
            Tangle x0 = x_[x_offset + pos0];
            Tangle x1 = x_[x_offset + pos1];

            y_[y_offset + pos0] = x0 * cos__ - x1 * sin__;
            y_[y_offset + pos1] = x0 * sin__ + x1 * cos__;
        }
    }
}

#endif
