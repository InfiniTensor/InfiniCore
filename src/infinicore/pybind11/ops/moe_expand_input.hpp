#pragma once
#include "infinicore/ops/moe_expand_input.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace infinicore::ops {
inline void bind_moe_expand_input(py::module &m) {
    m.def("moe_expand_input_with_inv_pos_", &op::moe_expand_input_with_inv_pos_, py::arg("expand_states"), py::arg("expand_scales"), py::arg("hidden_states"), py::arg("inv_pos"), py::arg("top_k"), py::arg("group_size") = 128, py::arg("format") = 0);
}
} // namespace infinicore::ops
