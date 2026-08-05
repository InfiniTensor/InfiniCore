#pragma once

#include "infinicore/ops/fp8_indexer_logits.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {
inline void bind_fp8_indexer_logits(py::module &m) {
    m.def("fp8_indexer_logits_",
          &op::fp8_indexer_logits_,
          py::arg("logits"),
          py::arg("q_fp8"),
          py::arg("kv_cache"),
          py::arg("block_tables"),
          py::arg("weights_fp32"),
          py::arg("positions"),
          py::arg("request_ids"));
}
} // namespace infinicore::ops
