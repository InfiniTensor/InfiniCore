#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/binary_cross_entropy_with_logits.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_binary_cross_entropy_with_logits(py::module &m) {
    // 1. 绑定 out-of-place 接口: out = binary_cross_entropy_with_logits(...)
    m.def("binary_cross_entropy_with_logits",
          &op::binary_cross_entropy_with_logits,
          py::arg("input"),
          py::arg("target"),
          py::arg("weight") = Tensor(),      // 默认为空 Tensor
          py::arg("pos_weight") = Tensor(),  // 默认为空 Tensor
          py::arg("reduction") = "mean",         // 默认归约方式为平均值
          R"doc(Measures Binary Cross Entropy between target and output logits.

Args:
    input: Tensor of arbitrary shape as unnormalized scores (logits).
    target: Tensor of the same shape as input with values between 0 and 1.
    weight: Optional rescaling weight for each loss component.
    pos_weight: Optional weight for positive examples (must be broadcastable).
    reduction: Specfies the reduction to apply: 'none' | 'mean' | 'sum'.

Returns:
    A tensor representing the loss.
)doc");

    // 2. 绑定指定输出接口: binary_cross_entropy_with_logits_(out, ...)
    m.def("binary_cross_entropy_with_logits_",
          &op::binary_cross_entropy_with_logits_,
          py::arg("out"),
          py::arg("input"),
          py::arg("target"),
          py::arg("weight") = Tensor(),
          py::arg("pos_weight") = Tensor(),
          py::arg("reduction") = "mean",
          R"doc(Specified output version of binary_cross_entropy_with_logits.

Args:
    out: The destination tensor to store the loss.
    input: Logits tensor.
    target: Target tensor.
    weight: Optional sample weight.
    pos_weight: Optional positive class weight.
    reduction: Specfies the reduction to apply.
)doc");
}

} // namespace infinicore::ops