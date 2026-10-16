#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/fp8_e4m3_dequant.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_fp8_e4m3_dequant(py::module &m) {
    m.def("fp8_e4m3_dequant_",
          &op::fp8_e4m3_dequant_,
          py::arg("output"),
          py::arg("input"),
          py::arg("scales"));
}

} // namespace infinicore::ops
