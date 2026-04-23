// python/infinicore/ops/split/split.cpp
#include <torch/extension.h>
torch::Tensor split_cuda(const torch::Tensor& input, int64_t split_size, int64_t dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("split", &split_cuda, "split", py::arg("input"), py::arg("split_size"), py::arg("dim")=0);
}