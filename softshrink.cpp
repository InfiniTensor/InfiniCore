// python/infinicore/ops/softshrink/softshrink.cpp
#include <torch/extension.h>
torch::Tensor softshrink(const torch::Tensor& input, float lambda);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softshrink", &softshrink, "softshrink", py::arg("input"), py::arg("lambda")=0.5f);
}