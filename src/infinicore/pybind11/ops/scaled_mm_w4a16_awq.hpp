#pragma once

#include "infinicore/ops/scaled_mm_w4a16_awq.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore {
inline void bind_scaled_mm_w4a16_awq(py::module &m) {
    m.def("scaled_mm_w4a16_awq", &op::scaled_mm_w4a16_awq,
          py::arg("input"), py::arg("qweight"), py::arg("qzeros"),
          py::arg("scales"), py::arg("bias") = std::nullopt);
    m.def("scaled_mm_w4a16_awq_", &op::scaled_mm_w4a16_awq_,
          py::arg("out"), py::arg("input"), py::arg("qweight"),
          py::arg("qzeros"), py::arg("scales"), py::arg("bias") = std::nullopt);
}
} // namespace infinicore
