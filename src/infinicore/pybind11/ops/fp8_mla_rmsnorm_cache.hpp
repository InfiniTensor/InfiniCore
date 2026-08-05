#pragma once

#include "infinicore/ops/fp8_mla_rmsnorm_cache.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {
inline void bind_fp8_mla_rmsnorm_cache(py::module &m) {
    m.def("fp8_mla_rmsnorm_cache_",
          &op::fp8_mla_rmsnorm_cache_,
          py::arg("cache"),
          py::arg("compressed_kv"),
          py::arg("norm_weight"),
          py::arg("rope"),
          py::arg("slot_mapping"),
          py::arg("eps"));
    m.def("fp8_mla_rmsnorm_dual_cache_",
          &op::fp8_mla_rmsnorm_dual_cache_,
          py::arg("cache"),
          py::arg("vendor_cache"),
          py::arg("compressed_kv"),
          py::arg("norm_weight"),
          py::arg("rope"),
          py::arg("slot_mapping"),
          py::arg("eps"));
}
} // namespace infinicore::ops
