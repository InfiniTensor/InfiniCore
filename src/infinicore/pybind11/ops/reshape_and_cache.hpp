#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/reshape_and_cache.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_reshape_and_cache(py::module &m) {
    m.def("reshape_and_cache",
          &op::reshape_and_cache,
          py::arg("key"),
          py::arg("value"),
          py::arg("key_cache"),
          py::arg("value_cache"),
          py::arg("slot_mapping"),
          py::arg("kv_cache_dtype"),
          py::arg("k_scale"),
          py::arg("v_scale"),
          R"doc(Reshape and cache key/value tensors into paged KV cache.)doc");
}

} // namespace infinicore::ops
