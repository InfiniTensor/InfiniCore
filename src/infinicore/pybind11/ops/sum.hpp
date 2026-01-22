#pragma once

#include "infinicore/ops/sum.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_sum(py::module &m) {
    m.def("sum_reduce",
          &op::sum_reduce,
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim") = false,
          R"doc(Reduces the input tensor along the specified dimension by taking the sum.)doc");

    m.def("sum_reduce_",
          &op::sum_reduce_,
          py::arg("input"),
          py::arg("output"),
          py::arg("dim"),
          py::arg("keepdim") = false,
          R"doc(In-place sum reduction along the specified dimension.)doc");

    m.def("sum_global",
          &op::sum_global,
          py::arg("input"),
          R"doc(Reduces the input tensor globally by taking the sum across all elements.)doc");

    m.def("sum_global_",
          &op::sum_global_,
          py::arg("input"),
          py::arg("output"),
          R"doc(In-place global sum reduction.)doc");
}

} // namespace infinicore::ops