#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/fused_moe.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline std::optional<Tensor> optional_tensor(py::object obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<Tensor>();
}

inline Tensor py_fused_moe(Tensor input,
                           Tensor token_selected_experts,
                           Tensor token_final_scales,
                           Tensor w1,
                           Tensor w2,
                           py::object b1,
                           py::object b2,
                           int activation) {
    return op::fused_moe(input, token_selected_experts, token_final_scales, w1, w2,
                         optional_tensor(b1), optional_tensor(b2),
                         static_cast<op::FusedMoeActivation>(activation));
}

inline void py_fused_moe_(Tensor out,
                          Tensor input,
                          Tensor token_selected_experts,
                          Tensor token_final_scales,
                          Tensor w1,
                          Tensor w2,
                          py::object b1,
                          py::object b2,
                          int activation) {
    op::fused_moe_(out, input, token_selected_experts, token_final_scales, w1, w2,
                   optional_tensor(b1), optional_tensor(b2),
                   static_cast<op::FusedMoeActivation>(activation));
}

inline void bind_fused_moe(py::module &m) {
    m.def("fused_moe",
          &ops::py_fused_moe,
          py::arg("input"),
          py::arg("token_selected_experts"),
          py::arg("token_final_scales"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("b1") = py::none(),
          py::arg("b2") = py::none(),
          py::arg("activation") = 1,
          R"doc(Fused MoE consuming topksoftmax values/indices. activation: 0=silu, 1=swiglu.)doc");

    m.def("fused_moe_",
          &ops::py_fused_moe_,
          py::arg("out"),
          py::arg("input"),
          py::arg("token_selected_experts"),
          py::arg("token_final_scales"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("b1") = py::none(),
          py::arg("b2") = py::none(),
          py::arg("activation") = 1,
          R"doc(In-place fused MoE consuming topksoftmax values/indices.)doc");
}

} // namespace infinicore::ops
