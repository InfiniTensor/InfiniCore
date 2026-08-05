#pragma once

#include "infinicore/ops/fused_rotary_embedding.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore {

inline void bind_fused_rotary_embedding(py::module &m) {
    m.def("fused_rotary_embedding_",
          &op::fused_rotary_embedding_,
          py::arg("query"),
          py::arg("key"),
          py::arg("positions"),
          py::arg("head_size"),
          py::arg("cos_sin_cache"),
          py::arg("is_neox"));
}

} // namespace infinicore
