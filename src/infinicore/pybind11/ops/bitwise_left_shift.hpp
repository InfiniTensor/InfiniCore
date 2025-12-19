#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/bitwise_left_shift.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_bitwise_left_shift(py::module &m) {
    m.def("bitwise_left_shift",
          &op::bitwise_left_shift,
          py::arg("a"),
          py::arg("b"),
          R"doc(Element-wise bitwise left shift of two tensors.)doc");

    m.def("bitwise_left_shift_",
          &op::bitwise_left_shift_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place element-wise tensor bitwise left shift.)doc");
}

} // namespace infinicore::ops
