#pragma once

#include "infinicore/ops/fused_moe_w4a8.hpp"
#include <pybind11/pybind11.h>

namespace infinicore::ops {

inline Tensor py_fused_moe_w4a8(
    Tensor input,
    Tensor selected_experts,
    Tensor routing_weights,
    Tensor w13_packed,
    Tensor w13_scale,
    Tensor w2_packed,
    Tensor w2_scale,
    int activation,
    bool weights_are_aiter_shuffled) {
    return op::fused_moe_w4a8(
        input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale,
        static_cast<op::FusedMoeActivation>(activation),
        weights_are_aiter_shuffled);
}

inline void py_fused_moe_w4a8_(
    Tensor output,
    Tensor input,
    Tensor selected_experts,
    Tensor routing_weights,
    Tensor w13_packed,
    Tensor w13_scale,
    Tensor w2_packed,
    Tensor w2_scale,
    int activation,
    bool weights_are_aiter_shuffled) {
    op::fused_moe_w4a8_(
        output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale,
        static_cast<op::FusedMoeActivation>(activation),
        weights_are_aiter_shuffled);
}

inline void bind_fused_moe_w4a8(pybind11::module &m) {
    m.def("fused_moe_w4a8",
          &py_fused_moe_w4a8,
          pybind11::arg("input"),
          pybind11::arg("selected_experts"),
          pybind11::arg("routing_weights"),
          pybind11::arg("w13_packed"),
          pybind11::arg("w13_scale"),
          pybind11::arg("w2_packed"),
          pybind11::arg("w2_scale"),
          pybind11::arg("activation") = 1,
          pybind11::arg("weights_are_aiter_shuffled") = false);
    m.def("fused_moe_w4a8_",
          &py_fused_moe_w4a8_,
          pybind11::arg("output"),
          pybind11::arg("input"),
          pybind11::arg("selected_experts"),
          pybind11::arg("routing_weights"),
          pybind11::arg("w13_packed"),
          pybind11::arg("w13_scale"),
          pybind11::arg("w2_packed"),
          pybind11::arg("w2_scale"),
          pybind11::arg("activation") = 1,
          pybind11::arg("weights_are_aiter_shuffled") = false);
}

} // namespace infinicore::ops
