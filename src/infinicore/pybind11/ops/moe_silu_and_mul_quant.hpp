#pragma once
#include "infinicore/ops/moe_silu_and_mul_quant.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace infinicore::ops {
inline void bind_moe_silu_and_mul_quant(py::module &m) {
    m.def("moe_silu_and_mul_quant_", &op::moe_silu_and_mul_quant_, py::arg("output"), py::arg("output_scale"), py::arg("input"), py::arg("format") = 0);
}
} // namespace infinicore::ops
