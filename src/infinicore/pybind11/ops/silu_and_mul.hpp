#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/silu_and_mul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_silu_and_mul(py::module &m) {
    // 绑定非原地函数: Tensor silu_and_mul(Tensor input)
    m.def("silu_and_mul",
          &op::silu_and_mul,
          py::arg("input"),
          R"doc(
          SiLU and Mul (SwiGLU) activation function.
          Input should be [..., 2*d], output will be [..., d].
          )doc");

    // 绑定原地/指定输出函数: void silu_and_mul_(Tensor output, Tensor input)
    m.def("silu_and_mul_",
          &op::silu_and_mul_,
          py::arg("output"),
          py::arg("input"),
          R"doc(
          In-place or destination-specified SiLU and Mul (SwiGLU) activation function.
          )doc");
}

} // namespace infinicore::ops
