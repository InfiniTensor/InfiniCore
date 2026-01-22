#pragma once

#include "infinicore/ops/var_mean.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_var_mean(py::module &m) {
    m.def("var_mean_reduce",
          &op::var_mean_reduce,
          py::arg("input"),
          py::arg("dim"),
          py::arg("correction"),
          py::arg("keepdim") = false,
          R"doc(Returns the variance and mean of each row of the input tensor in the given dimension.)doc");

    m.def("var_mean_reduce_",
          &op::var_mean_reduce_,
          py::arg("input"),
          py::arg("out_var"),
          py::arg("out_mean"),
          py::arg("dim"),
          py::arg("correction"),
          py::arg("keepdim") = false,
          R"doc(In-place version of var_mean_reduce.)doc");

    m.def("var_mean_global",
          &op::var_mean_global,
          py::arg("input"),
          py::arg("correction"),
          R"doc(Returns the global variance and mean.)doc");

    m.def("var_mean_global_",
          &op::var_mean_global_,
          py::arg("input"),
          py::arg("out_var"),
          py::arg("out_mean"),
          py::arg("correction"),
          R"doc(In-place global version.)doc");
}

} // namespace infinicore::ops