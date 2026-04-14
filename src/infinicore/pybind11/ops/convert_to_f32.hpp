#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/convert_to_f32.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_convert_to_f32(py::module &m) {
    m.def("convert_to_f32",
          &op::convert_to_f32,
          py::arg("x"),
          R"doc(Cast the input tensor to float32 (elementwise).)doc");

    m.def("convert_to_f32_",
          &op::convert_to_f32_,
          py::arg("y"),
          py::arg("x"),
          R"doc(Cast the input tensor to float32 into ``y`` (elementwise).)doc");
}

} // namespace infinicore::ops
