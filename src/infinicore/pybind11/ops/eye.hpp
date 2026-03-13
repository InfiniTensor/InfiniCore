#pragma once

#include <optional>

#include <pybind11/pybind11.h>

#include "infinicore/ops/eye.hpp"

namespace py = pybind11;

namespace infinicore::ops {

namespace {

Tensor py_eye(size_t n, py::object m, const DataType &dtype, const Device &device) {
    std::optional<size_t> m_opt = m.is_none() ? std::nullopt : std::make_optional(m.cast<size_t>());
    return op::eye(n, m_opt, dtype, device);
}

} // namespace

inline void bind_eye(py::module &m) {
    m.def("eye",
          &py_eye,
          py::arg("n"),
          py::arg("m") = py::none(),
          py::arg("dtype") = DataType::F32,
          py::arg("device") = Device::cpu(),
          R"doc(
Create an identity matrix of shape (n, m).

Args:
    n: Number of rows.
    m: Number of columns. If not provided, defaults to n (square matrix).
    dtype: Data type of the tensor. Defaults to float32.
    device: Device to create the tensor on. Defaults to CPU.

Returns:
    A 2D tensor with ones on the diagonal and zeros elsewhere.
)doc");
}

} // namespace infinicore::ops
