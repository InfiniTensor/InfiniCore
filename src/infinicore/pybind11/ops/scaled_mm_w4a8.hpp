#pragma once
#include "infinicore/ops/scaled_mm_w4a8.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace infinicore {
inline void bind_scaled_mm_w4a8(py::module &m) {
    m.def("scaled_mm_w4a8", &op::scaled_mm_w4a8, py::arg("a"), py::arg("b"), py::arg("a_scales"), py::arg("b_scales"), py::arg("bias") = std::nullopt, py::arg("trans_weight") = false);
    m.def("scaled_mm_w4a8_", &op::scaled_mm_w4a8_, py::arg("out"), py::arg("a"), py::arg("b"), py::arg("a_scales"), py::arg("b_scales"), py::arg("bias") = std::nullopt, py::arg("trans_weight") = false);
}
} // namespace infinicore
