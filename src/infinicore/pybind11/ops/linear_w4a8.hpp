#pragma once

#include "infinicore/ops/linear_w4a8.hpp"
#include <pybind11/pybind11.h>

namespace infinicore::ops {

inline Tensor py_linear_w4a8(Tensor input,
                             Tensor packed_weight,
                             Tensor weight_scale,
                             pybind11::object bias,
                             float alpha) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    return op::linear_w4a8(
        input, packed_weight, weight_scale, bias_tensor, alpha);
}

inline void py_linear_w4a8_(Tensor output,
                            Tensor input,
                            Tensor packed_weight,
                            Tensor weight_scale,
                            pybind11::object bias,
                            float alpha) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    op::linear_w4a8_(
        output, input, packed_weight, weight_scale, bias_tensor, alpha);
}

inline void bind_linear_w4a8(pybind11::module &m) {
    m.def("linear_w4a8",
          &py_linear_w4a8,
          pybind11::arg("input"),
          pybind11::arg("packed_weight"),
          pybind11::arg("weight_scale"),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("alpha") = 1.0f);
    m.def("linear_w4a8_",
          &py_linear_w4a8_,
          pybind11::arg("output"),
          pybind11::arg("input"),
          pybind11::arg("packed_weight"),
          pybind11::arg("weight_scale"),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("alpha") = 1.0f);
}

} // namespace infinicore::ops
