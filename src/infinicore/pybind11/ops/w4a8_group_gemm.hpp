#pragma once
#include "infinicore/ops/w4a8_group_gemm.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace infinicore {
inline void bind_w4a8_group_gemm(py::module &m) {
    m.def("w4a8_group_gemm_", &op::w4a8_group_gemm_, py::arg("out"), py::arg("input"), py::arg("weight"), py::arg("input_scale"), py::arg("weight_scale"), py::arg("tokens_per_experts"), py::arg("sorted_token_ids") = std::nullopt, py::arg("bias") = std::nullopt, py::arg("trans_weight") = true, py::arg("is_decode") = false);
}
} // namespace infinicore
