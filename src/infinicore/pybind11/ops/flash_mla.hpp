#pragma once

#include "infinicore/ops/flash_mla.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {
inline void bind_flash_mla(py::module &m) {
  m.def("flash_mla_sparse_prefill_", &op::flash_mla_sparse_prefill_,
        py::arg("output"), py::arg("query"), py::arg("kv_cache"),
        py::arg("indices"), py::arg("topk_lens"), py::arg("scale"));
  m.def("flash_mla_sparse_decode_", &op::flash_mla_sparse_decode_,
        py::arg("output"), py::arg("query"), py::arg("kv_cache"),
        py::arg("indices"), py::arg("topk_lens"), py::arg("scale"));
}
} // namespace infinicore::ops
