#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/vocab_parallel_embedding.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_vocab_parallel_embedding(py::module &m) {
    m.def("vocab_parallel_embedding_",
          &op::vocab_parallel_embedding_,
          py::arg("output"),
          py::arg("indices"),
          py::arg("weight"),
          py::arg("vocab_start"),
          py::arg("vocab_end"),
          R"doc(Embedding lookup for a tensor-parallel vocabulary shard.)doc");
}

} // namespace infinicore::ops
