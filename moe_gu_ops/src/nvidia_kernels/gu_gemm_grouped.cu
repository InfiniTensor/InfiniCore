#include "cuda_utils.h"
#include "nvidia_kernels_moe.h"
#include <vector>

// 引入 CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/include/cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/include/cutlass/gemm/kernel/default_gemm_grouped.h"

// ========================================================================
// 1. 定义 CUTLASS GEMM 类型
// ========================================================================

// 设定精度
using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;
using ElementAccumulator = float;

// 设定布局 (关键！)
// A (Input): RowMajor [M, K]
using LayoutA = cutlass::layout::RowMajor; 
// B (Weight): ColumnMajor [K, N] -> 物理上对应 RowMajor [N, K]
using LayoutB = cutlass::layout::ColumnMajor; 
// C (Output): RowMajor [M, N]
using LayoutC = cutlass::layout::RowMajor;

// 定义 Grouped GEMM 算子
// 架构: Sm80 (Ampere A100/3090), 如果是 V100 用 Sm70
using GemmGrouped = cutlass::gemm::device::GemmGrouped<
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::gemm::GemmGroupedIteratorAlgorithm::kOffsetBased,
    cutlass::arch::OpClassSimt, // FP32 使用 SIMT, 如果是 BF16/FP16 使用 OpClassTensorOp
    cutlass::arch::Sm80
>;

// ========================================================================
// 2. 参数准备 Kernel (Meta-Kernel)
// ========================================================================
// 这个 Kernel 负责在 GPU 上生成 CUTLASS 需要的参数结构体
__global__ void prepare_gemm_args(
    const int32_t* __restrict__ offsets,     // [Experts + 1]
    const float* __restrict__ input_base,    // 连续的 sorted_input
    const float* __restrict__ weight_base,   // 连续的权重 [E, N, K]
    float* __restrict__ output_base,         // 连续的 output
    cutlass::gemm::GemmCoord* problem_sizes, // 输出: [E] 尺寸
    const float** ptr_A,                     // 输出: [E] 指针
    const float** ptr_B,                     // 输出: [E] 指针
    float** ptr_C,                           // 输出: [E] 指针
    float** ptr_D,                           // 输出: [E] 指针
    int num_experts,
    int n, int k,         // N, K 是固定的
    int lda, int ldb, int ldc // Strides
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_experts) return;

    // 1. 算出当前专家的 M (Token 数)
    int start_row = offsets[idx];
    int end_row = offsets[idx + 1];
    int m = end_row - start_row;

    // 2. 填写尺寸: M, N, K
    problem_sizes[idx] = cutlass::gemm::GemmCoord(m, n, k);

    if (m > 0) {
        // 3. 计算指针位置
        // A: Input [start_row, 0]
        ptr_A[idx] = input_base + start_row * lda; 
        
        // B: Weight [idx, 0, 0]
        // 物理上 Weight 是 [E, N, K]，每个专家占 N*K
        ptr_B[idx] = weight_base + idx * (long long)n * k; 
        
        // C/D: Output [start_row, 0]
        ptr_C[idx] = output_base + start_row * ldc;
        ptr_D[idx] = ptr_C[idx];
    } else {
        ptr_A[idx] = nullptr;
        ptr_B[idx] = nullptr;
        ptr_C[idx] = nullptr;
        ptr_D[idx] = nullptr;
    }
}

// ========================================================================
// 3. 辅助 Kernel: Activation (SiLU * Mul)
// ========================================================================
__global__ void silu_and_mul_kernel(
    float* __restrict__ gate_up_output, // [Total_M, 2 * Inter]
    int total_elements,                 // Total_M * Inter
    int inter_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int row = tid / inter_dim;
    int col = tid % inter_dim;

    // 内存布局: [Gate | Up] (RowMajor)
    // Gate 在前半部分，Up 在后半部分
    long long gate_idx = (long long)row * (2 * inter_dim) + col;
    long long up_idx   = gate_idx + inter_dim;

    float gate_val = gate_up_output[gate_idx];
    float up_val   = gate_up_output[up_idx];

    // SiLU Calculation: x / (1 + exp(-x))
    float silu_val = gate_val / (1.0f + expf(-gate_val));
    
    // In-place update: 把结果写回 Gate 的位置
    gate_up_output[gate_idx] = silu_val * up_val;
}

// ========================================================================
// 4. Host Launcher
// ========================================================================

void launch_moe_gemm_ffn(
    const float* sorted_input,
    const int32_t* expert_offsets,
    float* sorted_output,
    const float* gate_up_proj_base, // [Experts, 2*Inter, Hidden]
    const float* down_proj_base,    // [Experts, Hidden, Inter]
    int num_experts,
    int hidden_dim, // K for GEMM1, N for GEMM2
    int inter_dim,  // N/2 for GEMM1, K for GEMM2
    cudaStream_t stream
) {
    // -------------------------------------------------------------
    // Phase 1: 准备 CUTLASS 参数的显存
    // -------------------------------------------------------------
    // Grouped GEMM 需要传入指针数组。
    // 计算 Workspace 大小
    size_t size_coord = num_experts * sizeof(cutlass::gemm::GemmCoord);
    size_t size_ptr   = num_experts * sizeof(void*);
    // 我们需要两套参数：一套给 GateUp GEMM，一套给 Down GEMM
    // 为了简单，我们复用同一块显存，算完第一个再算第二个。
    size_t workspace_bytes = size_coord + 4 * size_ptr;

    // 【注意】这里先 malloc，工业级应从外部传入 Workspace
    char* d_args_buffer;
    CUDA_CHECK(cudaMallocAsync(&d_args_buffer, workspace_bytes, stream));

    // 指针切分
    cutlass::gemm::GemmCoord* d_problem_sizes = (cutlass::gemm::GemmCoord*)(d_args_buffer);
    const float** d_ptr_A = (const float**)(d_args_buffer + size_coord);
    const float** d_ptr_B = (const float**)(d_args_buffer + size_coord + size_ptr);
    float** d_ptr_C       = (float**)(d_args_buffer + size_coord + 2 * size_ptr);
    float** d_ptr_D       = (float**)(d_args_buffer + size_coord + 3 * size_ptr);

    // -------------------------------------------------------------
    // Phase 2: GEMM 1 (Input * GateUp^T -> Middle)
    // -------------------------------------------------------------
    // Input: [M, Hidden]
    // Weight: [2*Inter, Hidden] (物理上) -> 逻辑上看作 [Hidden, 2*Inter] ColumnMajor
    // Output: [M, 2*Inter]
    
    // 我们需要一个 Middle Buffer 存 Gate+Up 的结果
    // 获取总 Token 数 (需要从 CPU 或者拷贝 offsets 的最后一个值，这里为了简单假设已知或同步获取)
    int total_tokens;
    CUDA_CHECK(cudaMemcpyAsync(&total_tokens, expert_offsets + num_experts, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // 等一下拿到 total_tokens

    float* d_middle;
    CUDA_CHECK(cudaMallocAsync(&d_middle, total_tokens * 2 * inter_dim * sizeof(float), stream));

    // 2.1 填充参数
    prepare_gemm_args<<< (num_experts+255)/256, 256, 0, stream >>>(
        expert_offsets,
        sorted_input,
        gate_up_proj_base,
        d_middle,
        d_problem_sizes, d_ptr_A, d_ptr_B, d_ptr_C, d_ptr_D,
        num_experts,
        2 * inter_dim, // N: 输出维度
        hidden_dim,    // K: 输入维度
        hidden_dim,    // lda (Input Row Stride)
        hidden_dim,    // ldb (Weight Column Stride = 物理上的 Row Stride) <--- 魔法在这里
        2 * inter_dim  // ldc (Output Row Stride)
    );

    // 2.2 运行 CUTLASS
    GemmGrouped gemm;
    typename GemmGrouped::Arguments args_1;
    args_1.problem_sizes = d_problem_sizes;
    args_1.count = num_experts;
    args_1.threadblock_count = 0;
    args_1.alpha = 1.0f; args_1.beta = 0.0f;
    args_1.ptr_A = d_ptr_A; args_1.ptr_B = d_ptr_B; args_1.ptr_C = d_ptr_C; args_1.ptr_D = d_ptr_D;
    // LDA/LDB/LDC 在 kernel 中算好了指针，这里设为 0 或默认即可，因为是指针模式
    
    // 初始化并运行
    size_t gemm_ws_size = gemm.get_workspace_size(args_1);
    void* gemm_ws = nullptr;
    if (gemm_ws_size > 0) CUDA_CHECK(cudaMallocAsync(&gemm_ws, gemm_ws_size, stream));
    
    CUDA_CHECK((cudaError_t)gemm.initialize(args_1, gemm_ws));
    CUDA_CHECK((cudaError_t)gemm.run(stream));

    // -------------------------------------------------------------
    // Phase 3: Activation (SiLU)
    // -------------------------------------------------------------
    int total_act_elements = total_tokens * inter_dim;
    silu_and_mul_kernel<<< (total_act_elements+255)/256, 256, 0, stream >>>(
        d_middle, total_act_elements, inter_dim
    );

    // -------------------------------------------------------------
    // Phase 4: GEMM 2 (Middle * Down^T -> Output)
    // -------------------------------------------------------------
    // Input (Middle): [M, Inter] (前半部分存了结果)
    // Weight: [Hidden, Inter] (物理上) -> 逻辑 [Inter, Hidden] ColumnMajor
    // Output: [M, Hidden]

    // 4.1 填充参数 (复用 d_args_buffer)
    prepare_gemm_args<<< (num_experts+255)/256, 256, 0, stream >>>(
        expert_offsets,
        d_middle,       // Input is now middle buffer
        down_proj_base,
        sorted_output,  // Final Output
        d_problem_sizes, d_ptr_A, d_ptr_B, d_ptr_C, d_ptr_D,
        num_experts,
        hidden_dim,     // N: 输出维度
        inter_dim,      // K: 输入维度
        2 * inter_dim,  // lda (Input Stride, 注意中间有 gap，stride 是 2*inter)
        inter_dim,      // ldb (Weight Stride)
        hidden_dim      // ldc (Output Stride)
    );

    // 4.2 运行 CUTLASS
    typename GemmGrouped::Arguments args_2 = args_1; // 复用配置，更新指针
    // 指针已经在 GPU 上更新了，所以 args 结构体里的指针不需要变，只需要重新 initialize
    // Wait... args结构体存的是 host 指针 d_problem_sizes。
    // 但是 CUTLASS 内部可能缓存了一些信息吗？最好重新构造 args。
    
    args_2.problem_sizes = d_problem_sizes;
    args_2.count = num_experts;
    args_2.ptr_A = d_ptr_A; args_2.ptr_B = d_ptr_B; args_2.ptr_C = d_ptr_C; args_2.ptr_D = d_ptr_D;
    
    CUDA_CHECK((cudaError_t)gemm.initialize(args_2, gemm_ws));
    CUDA_CHECK((cudaError_t)gemm.run(stream));

    // -------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------
    CUDA_CHECK(cudaFreeAsync(d_args_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(d_middle, stream));
    if (gemm_ws) CUDA_CHECK(cudaFreeAsync(gemm_ws, stream));
}