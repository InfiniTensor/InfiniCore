#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void softmaxKernel(
    Tdata *y_, const Tdata *x_,
    size_t othersize,   // = outer_size * inner_size
    size_t dimsize,     // = axis_size
    ptrdiff_t stride    // = inner_size
) {
    size_t other_idx = blockIdx.x;
    if (other_idx >= othersize) return;

    // -----------------------------------
    // 正确计算 softmax slice 的 base
    // -----------------------------------
    size_t inner_idx = other_idx % stride;
    size_t outer_idx = other_idx / stride;

    const Tdata *x = x_ + outer_idx * dimsize * stride + inner_idx;
    Tdata *y       = y_ + outer_idx * dimsize * stride + inner_idx;

    // ---------------------------
    // 1. block max
    // ---------------------------
    __shared__ Tcompute s_reduce[BLOCK_SIZE];
    __shared__ Tcompute s_max;

    Tcompute local_max = -INFINITY;

    for (size_t i = threadIdx.x; i < dimsize; i += BLOCK_SIZE) {
        Tcompute v = static_cast<Tcompute>(x[i * stride]);
        local_max = v > local_max ? v : local_max;
    }

    s_reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_reduce[threadIdx.x] =
                max(s_reduce[threadIdx.x], s_reduce[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) s_max = s_reduce[0];
    __syncthreads();

    // ---------------------------
    // 2. exp & sum
    // ---------------------------
    Tcompute local_sum = 0;

    for (size_t i = threadIdx.x; i < dimsize; i += BLOCK_SIZE) {
        Tcompute v =
            expf(static_cast<float>(x[i * stride]) - static_cast<float>(s_max));
        y[i * stride] = static_cast<Tdata>(v);
        local_sum += v;
    }

    s_reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
        }
        __syncthreads();
    }

    Tcompute sum = s_reduce[0];
    __syncthreads();

    // ---------------------------
    // 3. normalize
    // ---------------------------
    for (size_t i = threadIdx.x; i < dimsize; i += BLOCK_SIZE) {
        y[i * stride] =
            static_cast<Tdata>(
                static_cast<float>(y[i * stride]) / static_cast<float>(sum));
    }
}


#endif // __SOFTMAX_KERNEL_CUH__
