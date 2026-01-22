#pragma once

#include "infinicore/ops/all.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_all(py::module &m) {
    m.def("all_reduce",
          &op::all_reduce,
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim") = false,
          R"doc(Returns true if all elements in each row of the input tensor in the given dimension are true.)doc");

    m.def("all_reduce_",
          &op::all_reduce_,
          py::arg("input"),
          py::arg("output"),
          py::arg("dim"),
          py::arg("keepdim") = false,
          R"doc(In-place version of all_reduce.)doc");

    m.def("all_global",
          &op::all_global,
          py::arg("input"),
          R"doc(Returns true if all elements in the tensor are true.)doc");

    m.def("all_global_",
          &op::all_global_,
          py::arg("input"),
          py::arg("output"),
          R"doc(In-place global version.)doc");
}

} // namespace infinicore::ops