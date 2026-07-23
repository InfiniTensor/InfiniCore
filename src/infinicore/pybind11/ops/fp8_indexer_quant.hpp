#pragma once

#include "infinicore/ops/fp8_indexer_quant.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {
inline void bind_fp8_indexer_quant(py::module &m) {
    m.def("fp8_indexer_quant_",
          &op::fp8_indexer_quant_,
          py::arg("q_fp8"),
          py::arg("weights_fp32"),
          py::arg("q"),
          py::arg("weights"));
    m.def("fused_fp8_indexer_",
          &op::fused_fp8_indexer_,
          py::arg("q_fp8"),
          py::arg("weights_fp32"),
          py::arg("k_cache"),
          py::arg("q_raw"),
          py::arg("k_weights"),
          py::arg("norm_weight"),
          py::arg("norm_bias"),
          py::arg("positions"),
          py::arg("cos_sin_cache"),
          py::arg("slot_mapping"),
          py::arg("rope_dim"),
          py::arg("eps"),
          py::arg("weights_scale"));
}
} // namespace infinicore::ops
