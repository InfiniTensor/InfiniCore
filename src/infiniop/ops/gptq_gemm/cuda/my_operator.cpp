#include <torch/extension.h>
#include <pybind11/pybind11.h>

// 声明 CUDA 函数
torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit);


// 绑定到 Python
PYBIND11_MODULE(vllm_gptq, m) {
    m.def("gptq_gemm", &gptq_gemm, "GPTQ GEMM (CUDA)");
}
