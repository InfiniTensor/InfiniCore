#pragma once

#include "infinicore/ops/bucketize.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_bucketize(py::module &m) {
    m.def("bucketize",
          &op::bucketize,
          py::arg("input"),
          py::arg("boundaries"),
          py::arg("right") = false,
          R"doc(Returns the indices of the buckets to which each value in the input belongs.)doc");

    m.def("bucketize_",
          &op::bucketize_,
          py::arg("input"),
          py::arg("boundaries"),
          py::arg("output"),
          py::arg("right") = false,
          R"doc(In-place version of bucketize.)doc");
}

} // namespace infinicore::ops