#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/concat_and_cache_mla_int8.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_concat_and_cache_mla_int8(py::module &m) {
    m.def("concat_and_cache_mla_int8_",
          &op::concat_and_cache_mla_int8_,
          py::arg("kv_c_int8"),
          py::arg("kv_c_scale"),
          py::arg("k_pe_int8"),
          py::arg("k_pe_scale"),
          py::arg("kv_cache"),
          py::arg("kv_cache_scale"),
          py::arg("slot_mapping"),
          R"doc(In-place int8 MLA KV cache write via vllm_iluvatar on Iluvatar.)doc");
}

} // namespace infinicore::ops
