#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <random>
#include <float.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel: Find TopK elements using a simple parallel reduction (for demonstration)
__global__ void topk_kernel(const float* input, float* output, int* indices, int N, int K) {
    extern __shared__ float sdata[];
    float* sval = sdata;
    int* sind = (int*)&sval[blockDim.x];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    float val = (idx < N) ? input[idx] : -FLT_MAX;
    sval[tid] = val;
    sind[tid] = idx;
    __syncthreads();

    // Simple parallel reduction to find max K times
    for (int k = 0; k < K; ++k) {
        // Find max in block
        float max_val = -FLT_MAX;
        int max_idx = -1;
        for (int i = 0; i < blockDim.x; ++i) {
            if (sval[i] > max_val) {
                max_val = sval[i];
                max_idx = sind[i];
            }
        }
        if (tid == 0) {
            output[k] = max_val;
            indices[k] = max_idx;
        }
        __syncthreads();
        // Remove the found max for next iteration
        if (sval[tid] == max_val) sval[tid] = -FLT_MAX;
        __syncthreads();
    }
}

// Host TopK using CUDA
void cuda_topk(const float* h_input, float* h_output, int* h_indices, int N, int K) {
    float *d_input = nullptr, *d_output = nullptr;
    int *d_indices = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = 1; // For demonstration, use 1 block
    size_t shared_mem = threads * (sizeof(float) + sizeof(int));

    CUDA_CHECK(cudaEventRecord(start));
    topk_kernel<<<blocks, threads, shared_mem>>>(d_input, d_output, d_indices, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_indices, d_indices, K * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    const int N = 1024;
    const int K = 10;

    std::vector<float> h_input(N);
    std::vector<float> h_output(K);
    std::vector<int> h_indices(K);

    // Generate random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    for (int i = 0; i < N; ++i) {
        h_input[i] = dist(gen);
    }

    cuda_topk(h_input.data(), h_output.data(), h_indices.data(), N, K);

    printf("Top %d elements:\n", K);
    for (int i = 0; i < K; ++i) {
        printf("Value: %f, Index: %d\n", h_output[i], h_indices[i]);
    }

    return 0;
}