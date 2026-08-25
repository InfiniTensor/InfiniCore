#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/timestep_embedding.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_timestep_embedding(py::module &m) {
    m.def("timestep_embedding",
          &op::timestep_embedding,
          py::arg("timestep"),
          py::arg("embedding_dim") = 256,
          py::arg("max_period") = 10000.0f,
          R"doc(Build sinusoidal timestep embeddings on the active device.)doc");

    m.def("timestep_embedding_",
          &op::timestep_embedding_,
          py::arg("output"),
          py::arg("timestep"),
          py::arg("max_period") = 10000.0f,
          R"doc(Build sinusoidal timestep embeddings into output.)doc");
}

} // namespace infinicore::ops
