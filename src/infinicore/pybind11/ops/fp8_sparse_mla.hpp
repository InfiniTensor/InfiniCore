#pragma once

#include "infinicore/ops/fp8_sparse_mla.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {
inline void bind_fp8_sparse_mla(py::module &m) {
    m.def("fp8_sparse_mla_",
          &op::fp8_sparse_mla_,
          py::arg("output"),
          py::arg("query"),
          py::arg("kv_cache"),
          py::arg("indices"),
          py::arg("topk_lens"),
          py::arg("scale"));
}
} // namespace infinicore::ops
