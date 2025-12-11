// python/infinicore/ops/std/std.cpp
#include <torch/extension.h>
torch::Tensor std(const torch::Tensor& input, int64_t dim, bool unbiased, bool keepdim);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("std", &std, "std", py::arg("input"), py::arg("dim")=-1, py::arg("unbiased")=true, py::arg("keepdim")=false);
}