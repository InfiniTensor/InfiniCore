#ifndef __SWIGLU_CUDA_KERNEL_CUH__
#define __SWIGLU_CUDA_KERNEL_CUH__
template <typename T>
__device__ __forceinline__ T sigmoid(const T &x) {
    if constexpr (std::is_same_v<T, half2>) {
        return h2rcp(__hadd2(make_half2(1, 1), h2exp(__hneg2(x))));
    } else if constexpr (std::is_same_v<T, half>) {
        return hrcp(__hadd(half(1.f), __float2half(__expf(__half2float(__hneg(x))))));
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
    const size_t *shape,
    const ptrdiff_t *c_strides,
    const ptrdiff_t *a_strides,
    const ptrdiff_t *b_strides,
    int ndim) {
    int ind_c = 0;
    int ind_a = 0;
    int ind_b = 0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < length) {
        for (int j = ndim - 1; j >= 0; j--) {
            ind_c += (tid % (int)shape[j]) * (int)c_strides[j];
            ind_a += (tid % (int)shape[j]) * (int)a_strides[j];
            ind_b += (tid % (int)shape[j]) * (int)b_strides[j];
            tid = tid / (int)shape[j];
        }
        T gate = b[ind_b];
        T up = a[ind_a];
        if constexpr (std::is_same_v<T, half2>) {
            c[ind_c] = __hmul2(__hmul2(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, half>) {
            c[ind_c] = __hmul(__hmul(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            cuda_bfloat162 sig = sigmoid(gate);
            float gate0 = __bfloat162float(__low2bfloat16(gate));
            float gate1 = __bfloat162float(__high2bfloat16(gate));
            float sig0 = __bfloat162float(__low2bfloat16(sig));
            float sig1 = __bfloat162float(__high2bfloat16(sig));
            float up0 = __bfloat162float(__low2bfloat16(up));
            float up1 = __bfloat162float(__high2bfloat16(up));
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
