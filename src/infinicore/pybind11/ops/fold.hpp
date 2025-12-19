#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/fold.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_fold(py::module &m) {
    m.def("fold",
          &op::fold,
          py::arg("input"),
          py::arg("output_size"),
          py::arg("kernel_size"),
          py::arg("dilation") = std::make_tuple(1, 1),
          py::arg("padding") = std::make_tuple(0, 0),
          py::arg("stride") = std::make_tuple(1, 1),
          R"doc(Folds a tensor.)doc");
}

} // namespace infinicore::ops
