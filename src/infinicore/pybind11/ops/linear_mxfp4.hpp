#pragma once

#include "infinicore/ops/linear_mxfp4.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline Tensor py_linear_mxfp4(Tensor input,
                              Tensor packed_weight,
                              Tensor weight_scale,
                              py::object bias,
                              float alpha) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    return op::linear_mxfp4(input, packed_weight, weight_scale, bias_tensor, alpha);
}

inline void py_linear_mxfp4_(Tensor output,
                             Tensor input,
                             Tensor packed_weight,
                             Tensor weight_scale,
                             py::object bias,
                             float alpha) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    op::linear_mxfp4_(output, input, packed_weight, weight_scale, bias_tensor, alpha);
}

inline void bind_linear_mxfp4(py::module &m) {
    m.def("linear_mxfp4",
          &ops::py_linear_mxfp4,
          py::arg("input"),
          py::arg("packed_weight"),
          py::arg("weight_scale"),
          py::arg("bias") = py::none(),
          py::arg("alpha") = 1.0f);
    m.def("linear_mxfp4_",
          &ops::py_linear_mxfp4_,
          py::arg("output"),
          py::arg("input"),
          py::arg("packed_weight"),
          py::arg("weight_scale"),
          py::arg("bias") = py::none(),
          py::arg("alpha") = 1.0f);
}

} // namespace infinicore::ops
