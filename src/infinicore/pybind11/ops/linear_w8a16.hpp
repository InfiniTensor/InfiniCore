#pragma once

#include "infinicore/ops/linear_w8a16.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_linear_w8a16(Tensor input,
                       Tensor weight,
                       Tensor scale,
                       py::object bias) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    return op::linear_w8a16(input, weight, scale, bias_tensor);
}

void py_linear_w8a16_(Tensor out,
                      Tensor input,
                      Tensor weight,
                      Tensor scale,
                      py::object bias) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    op::linear_w8a16_(out, input, weight, scale, bias_tensor);
}

inline void bind_linear_w8a16(py::module &m) {
    m.def("linear_w8a16",
          &ops::py_linear_w8a16,
          py::arg("input"),
          py::arg("weight"),
          py::arg("scale"),
          py::arg("bias") = py::none(),
          R"doc(Weight-only int8 linear GEMM with BF16/FP16 activation.)doc");

    m.def("linear_w8a16_",
          &ops::py_linear_w8a16_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("scale"),
          py::arg("bias") = py::none(),
          R"doc(In-place weight-only int8 linear GEMM.)doc");}

} // namespace infinicore::ops
