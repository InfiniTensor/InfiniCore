#pragma once
#include "infinicore/ops/mamba2_scan.hpp"
#include <pybind11/pybind11.h>
namespace infinicore::ops {
inline void bind_mamba2_scan(pybind11::module &m) {
    m.def("mamba2_scan", &op::mamba2_scan, pybind11::arg("x"), pybind11::arg("dt"), pybind11::arg("b"), pybind11::arg("c"), pybind11::arg("a"), pybind11::arg("d"), pybind11::arg("dt_bias"), pybind11::arg("state"), pybind11::arg("offsets"), pybind11::arg("initial_indices"), pybind11::arg("final_indices"));
}
} // namespace infinicore::ops
