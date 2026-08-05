#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/bmm_strided.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_bmm_strided(py::module &m) {
    m.def("bmm_strided_",
          &op::bmm_strided_,
          py::arg("output"),
          py::arg("a"),
          py::arg("b"),
          R"doc(Batched matrix multiplication into a possibly strided output tensor.)doc");
}

} // namespace infinicore::ops
