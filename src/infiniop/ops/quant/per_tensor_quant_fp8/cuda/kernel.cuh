#ifndef __PER_TENSOR_QUANT_FP8_KERNEL_CUH__
#define __PER_TENSOR_QUANT_FP8_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define FULL_MASK 0xffffffff

// FP8 e4m3fn format: 1 sign + 4 exponent (bias 7) + 3 mantissa.
// Finite range: [-448, 448], min normal 2^-6, min subnormal 2^-9.
#define FP8_E4M3_MAX 448.0f

// warp reduce max
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, offset));
    }
    return val;
}

// float atomic max (safe version)
__device__ __forceinline__ void atomicMaxFloat(float *addr, float val) {
    int *addr_i = (int *)addr;
    int old = *addr_i;
    int assumed;

    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        float new_f = fmaxf(val, old_f);

        old = atomicCAS(addr_i, assumed, __float_as_int(new_f));

    } while (assumed != old);
}

__device__ inline int round_half_away_from_zero(float x) {
    float ax = fabsf(x);
    float r = floorf(ax + 0.5f);
    return (x >= 0.0f) ? (int)r : -(int)r;
}

// Convert a float to an FP8 e4m3fn byte (round half away from zero, saturate).
__device__ __forceinline__ unsigned char float_to_e4m3(float x) {
    unsigned int sign = 0;
    if (x < 0.0f) {
        sign = 0x80u;
        x = -x;
    }
    if (x >= FP8_E4M3_MAX) {
        return (unsigned char)(sign | 0x7eu); // 448 -> exp 15, mantissa 6
    }
    if (x < 0x1p-9f) {
        return (unsigned char)sign; // rounds to zero (0x1p-9 = 2^-9)
    }
    float e = floorf(log2f(x));
    int stored = (int)e + 7;
    if (stored <= 0) {
        // subnormal: value = m * 2^-9
        int m = (int)roundf(x * 512.0f); // x / 2^-9
        if (m > 7) {
            m = 7;
        }
        return (unsigned char)(sign | (unsigned int)m);
    }
    float frac = x * exp2f(-e) - 1.0f;
    int m = (int)roundf(frac * 8.0f);
    if (m == 8) {
        m = 0;
        stored += 1;
        if (stored > 15) {
            return (unsigned char)(sign | 0x7eu);
        }
    }
    return (unsigned char)(sign | ((unsigned int)stored << 3) | (unsigned int)m);
}

template <typename Tdata, unsigned int BLOCK_SIZE>
__device__ void perTensorAbsmaxFp8Kernel(float *x_scale, const Tdata *x,
                                         size_t batch_size, size_t channel, size_t hidden_dim, size_t width,
                                         ptrdiff_t strides_0, ptrdiff_t strides_1, ptrdiff_t strides_2, ptrdiff_t strides_3,
                                         int num_elements) {
    int idx = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + idx;
    int grid_size = blockDim.x * gridDim.x;

    float local_max = 0.f;

    for (int ind = gid; ind < num_elements; ind += grid_size) {
        int tid = ind;
        int w = tid % (int)width;
        tid = tid / (int)width;

        int h = tid % (int)hidden_dim;
        tid = tid / (int)hidden_dim;

        int c = tid % (int)channel;
        tid = tid / (int)channel;

        int b = tid % (int)batch_size;

        int index = w * (int)strides_3 + h * (int)strides_2 + c * (int)strides_1 + b * (int)strides_0;

        float v = fabsf((float)x[index]);

        local_max = fmaxf(local_max, v);
    }

    local_max = warpReduceMax(local_max);
    if ((idx & (WARP_SIZE - 1)) == 0) {
        atomicMaxFloat(x_scale, local_max / FP8_E4M3_MAX);
    }
}

template <typename Tdata, unsigned int BLOCK_SIZE>
__device__ void perTensorQuantFp8SymKernel(
    unsigned char *x_packed, float *x_scale, const Tdata *x,
    size_t batch_size, size_t channel, size_t hidden_dim, size_t width,
    ptrdiff_t strides_0, ptrdiff_t strides_1, ptrdiff_t strides_2, ptrdiff_t strides_3,
    ptrdiff_t p_strides_0, ptrdiff_t p_strides_1, ptrdiff_t p_strides_2, ptrdiff_t p_strides_3,
    int num_elements) {

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;

    float scale_val = 1.0f / x_scale[0];

    for (int ind = gid; ind < num_elements; ind += grid_size) {
        int tid = ind;
        int w = tid % (int)width;
        tid = tid / (int)width;

        int h = tid % (int)hidden_dim;
        tid = tid / (int)hidden_dim;

        int c = tid % (int)channel;
        tid = tid / (int)channel;

        int b = tid % (int)batch_size;

        int index = w * (int)strides_3 + h * (int)strides_2 + c * (int)strides_1 + b * (int)strides_0;
        int p_index = w * (int)p_strides_3 + h * (int)p_strides_2 + c * (int)p_strides_1 + b * (int)p_strides_0;

        float qf = (float)x[index] * scale_val;
        x_packed[p_index] = float_to_e4m3(qf);
    }
}

#endif // __PER_TENSOR_QUANT_FP8_KERNEL_CUH__
