#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/swiglu_cuda.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_swiglu_cuda(py::module &m) {
    m.def("swiglu_cuda",
          &op::swiglu_cuda,
          py::arg("a"),
          py::arg("b"),
          R"doc(SwiGLU CUDA activation function.)doc");

    m.def("swiglu_cuda_",
          &op::swiglu_cuda_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place SwiGLU CUDA activation function.)doc");
}

} // namespace infinicore::ops
