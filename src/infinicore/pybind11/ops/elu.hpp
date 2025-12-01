#pragma once

#include "infinicore/ops/elu.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {
inline void bind_elu(py::module &m) {
    m.def("elu", &op::elu, py::arg("input"), py::arg("alpha") = 1.0f,
          R"doc(Element-wise ELU activation function.

Args:
    input: Input tensor
    alpha: ELU parameter (default: 1.0)

Returns:
    Output tensor with ELU applied element-wise.
)doc");
    m.def("elu_", &op::elu_, py::arg("output"), py::arg("input"), py::arg("alpha") = 1.0f,
          R"doc(In-place ELU activation function.

Args:
    output: Output tensor (modified in-place)
    input: Input tensor
    alpha: ELU parameter (default: 1.0)
)doc");
}
} // namespace infinicore::ops