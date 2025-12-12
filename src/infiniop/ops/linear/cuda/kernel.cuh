#ifndef __LINEAR_KERNEL_CUH__
#define __LINEAR_KERNEL_CUH__

template <typename Tdata>
__device__ void postKernel(Tdata *y, int32_t *y_packed, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int M, int K, int N, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    int idx = row * N + col;
    float output1 = ((float)x_scale[row] * (float)w_scale[col] * ((float)y_packed[idx] + K * (float)x_zero[row] * (float)w_zero[col]));
    float output2 = 0.0f;
    float output3 = 0.0f;
    float tmp2 = (float)x_scale[row] * (float)w_scale[col] * (float)w_zero[col];
    float tmp3 = (float)x_scale[row] * (float)x_zero[row] * (float)w_scale[col];
    for (int ind = 0; ind < K; ind++) {
        output2 += tmp2 * (float)x_packed[row * K + ind];
        output3 += tmp3 * (float)w_packed[ind * N + col];
    }
    float output = alpha * (output1 - output2 - output3) + beta * (float)c[idx] + (float)bias[col];

    y[idx] = static_cast<Tdata>(output);
}

template <typename Tdata>
__device__ void postKernel(Tdata *y, int32_t *y_packed, const Tdata *c, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int M, int K, int N, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    int idx = row * N + col;
    float output1 = ((float)x_scale[row] * (float)w_scale[col] * ((float)y_packed[idx] + K * (float)x_zero[row] * (float)w_zero[col]));
    float output2 = 0.0f;
    float output3 = 0.0f;
    float tmp2 = (float)x_scale[row] * (float)w_scale[col] * (float)w_zero[col];
    float tmp3 = (float)x_scale[row] * (float)x_zero[row] * (float)w_scale[col];
    for (int ind = 0; ind < K; ind++) {
        output2 += tmp2 * (float)x_packed[row * K + ind];
        output3 += tmp3 * (float)w_packed[ind * N + col];
    }
    float output = alpha * (output1 - output2 - output3) + beta * (float)c[idx];

    y[idx] = static_cast<Tdata>(output);
}

template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const int8_t *w_packed, const Tdata *w_scale, int M, int K, int N, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    int idx = row * N + col;
    float output1 = (float)x_scale[row] * (float)w_scale[col] * ((float)y_packed[idx]);

    float output = alpha * output1 + beta * (float)c[idx] + (float)bias[col];

    y[idx] = static_cast<Tdata>(output);
}
template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const Tdata *c, const int8_t *x_packed, const Tdata *x_scale, const int8_t *w_packed, const Tdata *w_scale, int M, int K, int N, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    int idx = row * N + col;
    float output1 = (float)x_scale[row] * (float)w_scale[col] * ((float)y_packed[idx]);

    float output = alpha * output1 + beta * (float)c[idx];

    y[idx] = static_cast<Tdata>(output);
}
#endif // __LINEAR_KERNEL_CUH__
