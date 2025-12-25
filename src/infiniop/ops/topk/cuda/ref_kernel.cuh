#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <time.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// CPU top-k selection using std::sort (reference implementation)
void topk_cpu(const float* data, float* result, int n, int k) {
    float* temp = (float*)malloc(n * sizeof(float));
    memcpy(temp, data, n * sizeof(float));
    
    // Sort in descending order and take first k elements
    std::sort(temp, temp + n, std::greater<float>());
    
    memcpy(result, temp, k * sizeof(float));
    free(temp);
}

// Convert float to unsigned int for radix operations (preserving order)
__device__ __host__ inline unsigned int float_to_uint(float value) {
    unsigned int bits = *((unsigned int*)&value);
    // Handle sign bit: flip all bits if negative, otherwise just flip sign bit
    unsigned int mask = -((int)bits >> 31) | 0x80000000;
    return bits ^ mask;
}

// Convert unsigned int back to float
__device__ __host__ inline float uint_to_float(unsigned int bits) {
    unsigned int mask = ((bits >> 31) - 1) | 0x80000000;
    bits ^= mask;
    return *((float*)&bits);
}

// Helper function to get bit at position
__device__ __host__ inline unsigned int get_bit(float value, int bit_pos) {
    unsigned int bits = float_to_uint(value);
    return (bits >> bit_pos) & 1;
}

// GPU kernel: Count elements with bit=1 at given position
__global__ void count_bits_kernel(const float* data, int n, int bit_pos, int* count) {
    __shared__ int local_count[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Count locally
    local_count[tid] = 0;
    
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        if (get_bit(data[i], bit_pos) == 1) {
            local_count[tid]++;
        }
    }
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_count[tid] += local_count[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes result
    if (tid == 0) {
        atomicAdd(count, local_count[0]);
    }
}

// GPU kernel: Partition data based on bit value
__global__ void partition_kernel(const float* input, float* output_ones, float* output_zeros, 
                                 int n, int bit_pos, int* ones_idx, int* zeros_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float value = input[i];
        if (get_bit(value, bit_pos) == 1) {
            int pos = atomicAdd(ones_idx, 1);
            output_ones[pos] = value;
        } else {
            int pos = atomicAdd(zeros_idx, 1);
            output_zeros[pos] = value;
        }
    }
}

// Optimized GPU kernel: Partition with shared memory
// OPTIMIZATION: Two-phase approach using shared memory buffering
// Phase 1: Local partition to shared memory (fast)
// Phase 2: Batch write to global memory (coalesced)

// Radix select on GPU (iterative approach)
// ALGORITHM: Examine bits from MSB to LSB, partitioning at each step
// KEY INSIGHT: Elements with bit=1 are LARGER than elements with bit=0
void topk_gpu_radix(const float* d_input, float* d_output, int n, int k, bool optimized) {
    // Allocate working buffers
    float* d_current = nullptr;  // Current working set
    float* d_ones = nullptr;     // Elements with bit=1
    float* d_zeros = nullptr;    // Elements with bit=0
    int* d_ones_count = nullptr;
    int* d_zeros_count = nullptr;
    
    cudaMalloc(&d_current, n * sizeof(float));
    cudaMalloc(&d_ones, n * sizeof(float));
    cudaMalloc(&d_zeros, n * sizeof(float));
    cudaMalloc(&d_ones_count, sizeof(int));
    cudaMalloc(&d_zeros_count, sizeof(int));
    
    cudaMemcpy(d_current, d_input, n * sizeof(float), cudaMemcpyDeviceToDevice);
    
    int current_n = n;         // Number of elements in current working set
    int remaining_k = k;       // How many more elements we need to find
    int output_pos = 0;        // Position in output array
    
    // Process bits from MSB (31) to LSB (0)
    // Stop early if we've found all k elements or run out of data
    for (int bit_pos = 31; bit_pos >= 0 && remaining_k > 0 && current_n > 0; bit_pos--) {
        cudaMemset(d_ones_count, 0, sizeof(int));
        cudaMemset(d_zeros_count, 0, sizeof(int));
        
        dim3 block(BLOCK_SIZE);
        dim3 grid((current_n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (grid.x > 256) grid.x = 256;  // Limit blocks for better occupancy
        
        // PARTITION: Split current data by bit value at bit_pos
        // Elements with bit=1 are larger than elements with bit=0
        // if (optimized) {
        //     partition_kernel_optimized<<<grid, block>>>(d_current, d_ones, d_zeros, 
        //                                                 current_n, bit_pos, 
        //                                                 d_ones_count, d_zeros_count);
        // } else {
            partition_kernel<<<grid, block>>>(d_current, d_ones, d_zeros, 
                                              current_n, bit_pos, 
                                              d_ones_count, d_zeros_count);
        // }
        
        // Get partition sizes
        int ones_count, zeros_count;
        cudaMemcpy(&ones_count, d_ones_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&zeros_count, d_zeros_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        // DECISION: Which partition(s) contain the top-k elements?
        if (ones_count >= remaining_k) {
            // Case 1: All remaining top-k elements are in the ones partition
            // Discard zeros partition, continue searching in ones partition
            cudaMemcpy(d_current, d_ones, ones_count * sizeof(float), cudaMemcpyDeviceToDevice);
            current_n = ones_count;
        } else {
            // Case 2: Take ALL elements from ones partition (they're definitely top-k)
            // Continue searching for (remaining_k - ones_count) elements in zeros partition
            cudaMemcpy(d_output + output_pos, d_ones, ones_count * sizeof(float), 
                      cudaMemcpyDeviceToDevice);
            output_pos += ones_count;
            remaining_k -= ones_count;
            
            cudaMemcpy(d_current, d_zeros, zeros_count * sizeof(float), cudaMemcpyDeviceToDevice);
            current_n = zeros_count;
        }
    }
    
    // Copy any remaining elements (happens when bits exhausted or k elements found)
    if (remaining_k > 0 && current_n > 0) {
        int copy_count = min(remaining_k, current_n);
        cudaMemcpy(d_output + output_pos, d_current, copy_count * sizeof(float), 
                  cudaMemcpyDeviceToDevice);
    }
    
    // Cleanup
    cudaFree(d_current);
    cudaFree(d_ones);
    cudaFree(d_zeros);
    cudaFree(d_ones_count);
    cudaFree(d_zeros_count);
}

// Generate random float data
void generate_random_data(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 288888.0f - 480.0f;
    }
}

// Verify results
bool verify_topk(const float* cpu_result, const float* gpu_result, int k, float tolerance = 0.01f) {
    // Sort both results for comparison
    float* cpu_sorted = (float*)malloc(k * sizeof(float));
    float* gpu_sorted = (float*)malloc(k * sizeof(float));
    
    memcpy(cpu_sorted, cpu_result, k * sizeof(float));
    memcpy(gpu_sorted, gpu_result, k * sizeof(float));
    
    std::sort(cpu_sorted, cpu_sorted + k, std::greater<float>());
    std::sort(gpu_sorted, gpu_sorted + k, std::greater<float>());
    
    bool correct = true;
    int mismatches = 0;
    
    for (int i = 0; i < k; i++) {
        float diff = fabs(cpu_sorted[i] - gpu_sorted[i]);
        if (diff > tolerance) {
            if (mismatches < 10) {
                printf("Mismatch at %d: CPU=%.3f, GPU=%.3f (diff=%.3f)\n", 
                       i, cpu_sorted[i], gpu_sorted[i], diff);
            }
            mismatches++;
            correct = false;
        }
    }
    
    if (mismatches > 0) {
        printf("Total mismatches: %d / %d\n", mismatches, k);
    }
    
    free(cpu_sorted);
    free(gpu_sorted);
    
    return correct;
}

void print_array(const float* arr, int n, const char* name) {
    printf("%s (first 10): ", name);
    for (int i = 0; i < min(n, 10); i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main() {
    srand(42);
    
    const int N = 1024 * 1024; // s
    const int K = 50; 
    
    printf("Top-K Radix Select with Bit Comparison\n");
    printf("Data size: %d elements\n", N);
    printf("K: %d\n\n", K);
    
    // Allocate host memory
    float* h_data = (float*)malloc(N * sizeof(float));
    float* h_result_cpu = (float*)malloc(K * sizeof(float));
    float* h_result_gpu_naive = (float*)malloc(K * sizeof(float));
    float* h_result_gpu_opt = (float*)malloc(K * sizeof(float));
    
    // Generate random data
    printf("Generating random data...\n");
    generate_random_data(h_data, N);
    
    // Allocate device memory
    float* d_data;
    float* d_result_naive;
    float* d_result_opt;
    
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result_naive, K * sizeof(float));
    cudaMalloc(&d_result_opt, K * sizeof(float));
    
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // ==================== CPU Implementation ====================
    printf("\n==================== CPU Implementation ====================\n");
    clock_t start_cpu = clock();
    topk_cpu(h_data, h_result_cpu, N, K);
    clock_t end_cpu = clock();
    double time_cpu = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;
    
    printf("CPU time: %.3f ms\n", time_cpu);
    print_array(h_result_cpu, K, "CPU Top-K");
    
    // ==================== GPU Naive Implementation ====================
    printf("\n==================== GPU Naive Implementation ====================\n");
    
    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);
    
    cudaEventRecord(start_naive);
    topk_gpu_radix(d_data, d_result_naive, N, K, false);
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);
    
    float time_naive = 0;
    cudaEventElapsedTime(&time_naive, start_naive, stop_naive);
    
    cudaMemcpy(h_result_gpu_naive, d_result_naive, K * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("GPU Naive time: %.3f ms\n", time_naive);
    printf("Speedup vs CPU: %.2fx\n", time_cpu / time_naive);
    print_array(h_result_gpu_naive, K, "GPU Naive Top-K");
    
    if (verify_topk(h_result_cpu, h_result_gpu_naive, K)) {
        printf("✓ GPU Naive result is CORRECT\n");
    } else {
        printf("✗ GPU Naive result is INCORRECT\n");
    }
    
    // ==================== GPU Optimized Implementation ====================
    printf("\n==================== GPU Optimized Implementation ====================\n");
    
    cudaEvent_t start_opt, stop_opt;
    cudaEventCreate(&start_opt);
    cudaEventCreate(&stop_opt);
    
    cudaEventRecord(start_opt);
    topk_gpu_radix(d_data, d_result_opt, N, K, true);
    cudaEventRecord(stop_opt);
    cudaEventSynchronize(stop_opt);
    
    float time_opt = 0;
    cudaEventElapsedTime(&time_opt, start_opt, stop_opt);
    
    cudaMemcpy(h_result_gpu_opt, d_result_opt, K * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("GPU Optimized time: %.3f ms\n", time_opt);
    printf("Speedup vs CPU: %.2fx\n", time_cpu / time_opt);
    printf("Speedup vs Naive GPU: %.2fx\n", time_naive / time_opt);
    print_array(h_result_gpu_opt, K, "GPU Optimized Top-K");
    
    if (verify_topk(h_result_cpu, h_result_gpu_opt, K)) {
        printf("✓ GPU Optimized result is CORRECT\n");
    } else {
        printf("✗ GPU Optimized result is INCORRECT\n");
    }
    
    // ==================== Performance Summary ====================
    printf("\n==================== Performance Summary ====================\n");
    printf("Data size: %d elements (%.2f MB)\n", N, N * sizeof(float) / (1024.0 * 1024.0));
    printf("K: %d\n", K);
    printf("%-25s: %8.3f ms\n", "CPU (std::sort)", time_cpu);
    printf("%-25s: %8.3f ms (%.2fx speedup)\n", "GPU Naive", time_naive, time_cpu / time_naive);
    printf("%-25s: %8.3f ms (%.2fx speedup)\n", "GPU Optimized", time_opt, time_cpu / time_opt);
    printf("=============================================================\n");
    
    // Cleanup
    free(h_data);
    free(h_result_cpu);
    free(h_result_gpu_naive);
    free(h_result_gpu_opt);
    
    cudaFree(d_data);
    cudaFree(d_result_naive);
    cudaFree(d_result_opt);
    
    cudaEventDestroy(start_naive);
    cudaEventDestroy(stop_naive);
    cudaEventDestroy(start_opt);
    cudaEventDestroy(stop_opt);
    
    return 0;
}