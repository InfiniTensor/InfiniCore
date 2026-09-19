#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/lightning_attention.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_lightning_attention(py::module &m) {
    m.def("lightning_attention",
          &op::lightning_attention,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("slope"),
          py::arg("initial_state"),
          py::arg("initial_state_indices"),
          py::arg("final_state_indices"),
          R"doc(Indexed-pool lightning attention (MiniMax-01 style).
Returns out [B, T, H, D]; updates `initial_state` in place at final_state_indices rows.)doc");
}

} // namespace infinicore::ops
