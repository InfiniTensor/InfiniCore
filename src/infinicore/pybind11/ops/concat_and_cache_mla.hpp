#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/concat_and_cache_mla.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_concat_and_cache_mla(py::module &m) {
    m.def("concat_and_cache_mla_",
          &op::concat_and_cache_mla_,
          py::arg("kv_c"),
          py::arg("k_pe"),
          py::arg("kv_cache"),
          py::arg("slot_mapping"),
          py::arg("kv_cache_dtype"),
          py::arg("scale"),
          R"doc(In-place MLA KV concat and paged cache write via vllm_iluvatar on Iluvatar.)doc");
}

} // namespace infinicore::ops
