#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/addr.hpp"

namespace py = pybind11;
namespace infinicore::ops {
    inline void bind_addr(py::module_ &m) {
    m.def(
        "addr",
        &op::addr,
        py::arg("input"),
        py::arg("vec1"),
        py::arg("vec2"),
        py::arg("alpha"),
        py::arg("beta"),
        R"doc(Addr.)doc");
    m.def(
        "addr_",
        &op::addr_,
        py::arg("out"),
        py::arg("input"),
        py::arg("vec1"),
        py::arg("vec2"),
        py::arg("beta"),
        py::arg("alpha"),
        R"doc(Addr.)doc");
}
}