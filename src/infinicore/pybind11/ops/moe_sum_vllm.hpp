#pragma once
#include "infinicore/ops/moe_sum_vllm.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace infinicore::ops {
inline void bind_moe_sum_vllm(py::module &m) {
    m.def("moe_sum_vllm_", &op::moe_sum_vllm_, py::arg("output"), py::arg("input"), py::arg("topk_weights") = std::nullopt, py::arg("extra_residual") = std::nullopt, py::arg("routed_scale") = 1.0, py::arg("residual_scale") = 1.0);
}
} // namespace infinicore::ops
