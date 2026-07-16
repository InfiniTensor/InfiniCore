#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/moe_sum.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_moe_sum(py::module &m) {
    m.def("moe_sum",
          &op::moe_sum,
          py::arg("input"),
          R"doc(
          MoE top-k accumulate: input [M, topk, H] -> [M, H].
          Matches vLLM ``_custom_ops.moe_sum`` / sum over dim=1.
          )doc");

    m.def("moe_sum_",
          &op::moe_sum_,
          py::arg("output"),
          py::arg("input"),
          R"doc(
          Destination MoE top-k accumulate into ``output`` [M, H].
          )doc");
}

} // namespace infinicore::ops
