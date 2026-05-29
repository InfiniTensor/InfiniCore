#ifndef __SWIGLU_CUDA_KERNEL_CUH__
#define __SWIGLU_CUDA_KERNEL_CUH__

template <typename T>
__device__ __forceinline__ T sigmoid(const T &x) {
    if constexpr (std::is_same_v<T, half2>) {
        return h2rcp(__hadd2(make_half2(1, 1), h2exp(__hneg2(x))));
    } else if constexpr (std::is_same_v<T, half>) {
        // The original CUDA implementation's reliance on platform-specific intrinsics like hrcp for half-precision,
        // which was not supported on the MUSA platform.
        // To resolve this, the half-precision input is first converted to a higher-precision float,
        // the calculation is performed, and the result is cast back to half.
        float xf = __half2float(x);
        float sigf = 1.0f / (1.0f + std::exp(-xf));
        return __float2half(sigf);
    } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
        float x0 = __bfloat162float(__low2bfloat16(x));
        float x1 = __bfloat162float(__high2bfloat16(x));
        float sig0 = __frcp_rn(__fadd_rn(1.0f, __expf(-x0)));
        float sig1 = __frcp_rn(__fadd_rn(1.0f, __expf(-x1)));
        return __floats2bfloat162_rn(sig0, sig1);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        float xf = __bfloat162float(x);
        return __float2bfloat16_rn(__frcp_rn(__fadd_rn(1.0f, __expf(-xf))));
    } else if constexpr (std::is_same_v<T, float>) {
        return __frcp_rn(__fadd_rn(1, __expf(-x)));
    } else {
        return 1 / (1 + std::exp(-x));
    }
}

template <typename T, unsigned int BLOCK_SIZE>
__device__ void SwiGLUCudaKernel(
    T *c,
    const T *a,
    const T *b,
    int length,
    size_t batch, size_t seq_len, size_t hidden_dim,
    ptrdiff_t c_strides_0, ptrdiff_t c_strides_1, ptrdiff_t c_strides_2,
    ptrdiff_t a_strides_0, ptrdiff_t a_strides_1, ptrdiff_t a_strides_2,
    ptrdiff_t b_strides_0, ptrdiff_t b_strides_1, ptrdiff_t b_strides_2) {
    int ind_c = 0;
    int ind_a = 0;
    int ind_b = 0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < length) {
        ind_c += tid % (int)hidden_dim * (int)c_strides_2;
        ind_a += tid % (int)hidden_dim * (int)a_strides_2;
        ind_b += tid % (int)hidden_dim * (int)b_strides_2;
        tid = tid / (int)hidden_dim;
        ind_c += (tid % (int)seq_len) * (int)c_strides_1;
        ind_a += (tid % (int)seq_len) * (int)a_strides_1;
        ind_b += (tid % (int)seq_len) * (int)b_strides_1;
        tid = tid / (int)seq_len;
        ind_c += (tid % (int)batch) * (int)c_strides_0;
        ind_a += (tid % (int)batch) * (int)a_strides_0;
        ind_b += (tid % (int)batch) * (int)b_strides_0;

        T gate = b[ind_b];
        T up = a[ind_a];

        if constexpr (std::is_same_v<T, half2>) {
            c[ind_c] = __hmul2(__hmul2(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, half>) {
            c[ind_c] = __hmul(__hmul(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            cuda_bfloat162 sig = sigmoid(gate);
            // On the MUSA platform, `__low2float()` and `__high2float()` are used to directly
            // extract and convert bfloat16 values to float. These functions replace the
            // two-step process used in CUDA (e.g., `__low2bfloat16` followed by `__bfloat162float`).
            // Since MUSA may not support '__low2bfloat16'
            float gate0 = __low2float(gate);
            float gate1 = __high2float(gate);
            float sig0 = __low2float(sig);
            float sig1 = __high2float(sig);
            float up0 = __low2float(up);
            float up1 = __high2float(up);
            float res0 = __fmul_rn(__fmul_rn(gate0, sig0), up0);
            float res1 = __fmul_rn(__fmul_rn(gate1, sig1), up1);
            c[ind_c] = __floats2bfloat162_rn(res0, res1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 sig = sigmoid(gate);
            float gatef = __bfloat162float(gate);
            float sigf = __bfloat162float(sig);
            float upf = __bfloat162float(up);
            c[ind_c] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(gatef, sigf), upf));
        } else if constexpr (std::is_same_v<T, float>) {
            c[ind_c] = __fmul_rn(__fmul_rn(gate, sigmoid(gate)), up);
        } else {
            c[ind_c] = gate * sigmoid(gate) * up;
        }
    }
}

#endif // __SWIGLU_CUDA_KERNEL_CUH__
