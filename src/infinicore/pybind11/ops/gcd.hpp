#pragma once
#include "infinicore/ops/gcd.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_gcd(py::module &m) {
    m.def("gcd",
          &op::gcd,
          py::arg("input"),
          py::arg("other"));

    m.def("gcd_",
          &op::gcd_,
          py::arg("input"),
          py::arg("other"),
          py::arg("output"));
}

} // namespace infinicore::ops