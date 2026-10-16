#pragma once

#include "infinicore/ops/fused_moe_w8a8.hpp"
#include <pybind11/pybind11.h>

namespace infinicore::ops {

inline void bind_fused_moe_w8a8(pybind11::module &m) {
    m.def("fused_moe_w8a8",
          &op::fused_moe_w8a8,
          pybind11::arg("input"),
          pybind11::arg("selected_experts"),
          pybind11::arg("routing_weights"),
          pybind11::arg("w1"),
          pybind11::arg("w1_scale"),
          pybind11::arg("w2"),
          pybind11::arg("w2_scale"));
    m.def("fused_moe_w8a8_",
          &op::fused_moe_w8a8_,
          pybind11::arg("output"),
          pybind11::arg("input"),
          pybind11::arg("selected_experts"),
          pybind11::arg("routing_weights"),
          pybind11::arg("w1"),
          pybind11::arg("w1_scale"),
          pybind11::arg("w2"),
          pybind11::arg("w2_scale"));
}

} // namespace infinicore::ops
