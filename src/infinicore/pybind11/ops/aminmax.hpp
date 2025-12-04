#pragma once

#include "infinicore/ops/aminmax.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::ops {

std::pair<Tensor, Tensor> py_aminmax(Tensor input, py::object dim, bool keepdim) {
    std::optional<int64_t> dim_opt = std::nullopt;
    if (!dim.is_none()) {
        dim_opt = dim.cast<int64_t>();
    }
    return op::aminmax(input, dim_opt, keepdim);
}

void py_aminmax_(Tensor min_output, Tensor max_output, Tensor input, py::object dim, bool keepdim) {
    std::optional<int64_t> dim_opt = std::nullopt;
    if (!dim.is_none()) {
        dim_opt = dim.cast<int64_t>();
    }
    op::aminmax_(min_output, max_output, input, dim_opt, keepdim);
}

inline void bind_aminmax(py::module &m) {
    m.def("aminmax", &py_aminmax,
          py::arg("input"),
          py::arg("dim") = py::none(),
          py::arg("keepdim") = false,
          R"doc(Returns a tuple (min, max) of the minimum and maximum values of the input tensor.)doc");

    m.def("aminmax_", &py_aminmax_,
          py::arg("min_output"),
          py::arg("max_output"),
          py::arg("input"),
          py::arg("dim") = py::none(),
          py::arg("keepdim") = false,
          R"doc(In-place version of aminmax.)doc");
}

} // namespace infinicore::ops
