#pragma once

#include "infinicore/ops/fused_moe_mxfp4.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline Tensor py_fused_moe_mxfp4(Tensor input,
                                 Tensor selected_experts,
                                 Tensor routing_weights,
                                 Tensor w13_packed,
                                 Tensor w13_scale,
                                 Tensor w2_packed,
                                 Tensor w2_scale,
                                 int activation) {
    return op::fused_moe_mxfp4(
        input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale,
        static_cast<op::FusedMoeActivation>(activation));
}

inline void py_fused_moe_mxfp4_(Tensor output,
                                Tensor input,
                                Tensor selected_experts,
                                Tensor routing_weights,
                                Tensor w13_packed,
                                Tensor w13_scale,
                                Tensor w2_packed,
                                Tensor w2_scale,
                                int activation) {
    op::fused_moe_mxfp4_(
        output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale,
        static_cast<op::FusedMoeActivation>(activation));
}

inline void bind_fused_moe_mxfp4(py::module &m) {
    m.def("fused_moe_mxfp4",
          &ops::py_fused_moe_mxfp4,
          py::arg("input"),
          py::arg("selected_experts"),
          py::arg("routing_weights"),
          py::arg("w13_packed"),
          py::arg("w13_scale"),
          py::arg("w2_packed"),
          py::arg("w2_scale"),
          py::arg("activation") = 1);
    m.def("fused_moe_mxfp4_",
          &ops::py_fused_moe_mxfp4_,
          py::arg("output"),
          py::arg("input"),
          py::arg("selected_experts"),
          py::arg("routing_weights"),
          py::arg("w13_packed"),
          py::arg("w13_scale"),
          py::arg("w2_packed"),
          py::arg("w2_scale"),
          py::arg("activation") = 1);
}

} // namespace infinicore::ops
