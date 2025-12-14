#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/matrix_power.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_matrix_power(py::module &m) {
    m.def("matrix_power",
          &op::matrix_power,
          py::arg("input"),
          py::arg("n"),
          R"doc(Compute the n-th power of a square matrix.)doc");

    m.def("matrix_power_",
          &op::matrix_power_,
          py::arg("output"),
          py::arg("input"),
          py::arg("n"),
          R"doc(In-place matrix power computation.)doc");
}

} // namespace infinicore::ops

