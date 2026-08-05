#pragma once

#include "infinicore/ops/paged_attention_mla.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore {

inline void bind_paged_attention_mla(py::module &m) {
    m.def("paged_attention_mla_",
          &op::paged_attention_mla_,
          py::arg("output"),
          py::arg("query"),
          py::arg("kv_cache"),
          py::arg("scale"),
          py::arg("block_tables"),
          py::arg("context_lens"),
          py::arg("max_context_len"));
}

} // namespace infinicore
