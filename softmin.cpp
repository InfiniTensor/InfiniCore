// python/infinicore/ops/softmin/softmin.cpp
#include <torch/extension.h>
torch::Tensor softmin(const torch::Tensor& input);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmin", &softmin);
}