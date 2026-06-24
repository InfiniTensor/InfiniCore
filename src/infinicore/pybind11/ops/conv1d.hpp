#pragma once

#include <optional>
#include <pybind11/pybind11.h>

#include "infinicore/ops/conv1d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_conv1d(py::module &m) {
    m.def(
        "conv1d",
        [](::infinicore::Tensor input,
           ::infinicore::Tensor weight,
           std::optional<::infinicore::Tensor> bias,
           size_t stride,
           size_t padding,
           size_t dilation,
           size_t groups) {
            return op::conv1d(input, weight, bias, stride, padding, dilation, groups);
        },
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1,
        py::arg("groups") = 1,
        R"doc(Conv1d out-of-place.)doc");

    m.def(
        "conv1d_",
        [](::infinicore::Tensor output,
           ::infinicore::Tensor input,
           ::infinicore::Tensor weight,
           std::optional<::infinicore::Tensor> bias,
           size_t stride,
           size_t padding,
           size_t dilation,
           size_t groups) {
            op::conv1d_(output, input, weight, bias, stride, padding, dilation, groups);
        },
        py::arg("output"),
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1,
        py::arg("groups") = 1,
        R"doc(Conv1d in-place variant writing to provided output tensor.)doc");
}

} // namespace infinicore::ops
