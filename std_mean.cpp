// python/infinicore/ops/std_mean/std_mean.cpp
#include <torch/extension.h>
std::tuple<torch::Tensor, torch::Tensor> std_mean(const torch::Tensor& input);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("std_mean", &std_mean);
}