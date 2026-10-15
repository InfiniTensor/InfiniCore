#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/kimi_attention_residual.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_kimi_attention_residual(py::module &m) {
    m.def("kimi_attention_residual",
          &op::kimi_attention_residual,
          py::arg("values"),
          py::arg("norm_weight"),
          py::arg("projection_weight"),
          py::arg("epsilon") = 1e-6f);

    m.def("kimi_attention_residual_",
          &op::kimi_attention_residual_,
          py::arg("output"),
          py::arg("values"),
          py::arg("norm_weight"),
          py::arg("projection_weight"),
          py::arg("epsilon") = 1e-6f);
}

} // namespace infinicore::ops
