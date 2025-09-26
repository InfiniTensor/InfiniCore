#pragma once
#include "../../tensor/tensor.hpp"
#include "infinicore.hpp"
#include <pybind11/pybind11.h>

namespace infinicore::py {

Tensor matmul(const Tensor &a, const Tensor &b);
void matmul_(Tensor &c, const Tensor &a, const Tensor &b);

inline void bind_matmul(pybind11::module &m) {
    m.def("matmul",
          &matmul,
          pybind11::arg("a"),
          pybind11::arg("b"),
          R"doc(Matrix multiplication of two tensors.)doc");

    m.def("matmul_",
          &matmul_,
          pybind11::arg("c"),
          pybind11::arg("a"),
          pybind11::arg("b"),
          R"doc(In-place matrix multiplication.)doc");
}

} // namespace infinicore::py
