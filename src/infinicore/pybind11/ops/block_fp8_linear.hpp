#pragma once

#include "infinicore/ops/block_fp8_linear.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline Tensor py_block_fp8_linear(Tensor input,
                                  Tensor weight,
                                  Tensor weight_scale) {
    return op::block_fp8_linear(input, weight, weight_scale);
}

inline void py_block_fp8_linear_(Tensor output,
                                 Tensor input,
                                 Tensor weight,
                                 Tensor weight_scale) {
    op::block_fp8_linear_(output, input, weight, weight_scale);
}

inline void bind_block_fp8_linear(py::module &m) {
    m.def("block_fp8_linear",
          &ops::py_block_fp8_linear,
          py::arg("input"),
          py::arg("weight"),
          py::arg("weight_scale"));
    m.def("block_fp8_linear_",
          &ops::py_block_fp8_linear_,
          py::arg("output"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("weight_scale"));
}

} // namespace infinicore::ops
