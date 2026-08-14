#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/moe_fused_dense_ascend.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_moe_fused_dense_ascend(py::module &m) {
    m.def("moe_fused_dense_ascend",
          &op::moe_fused_dense_ascend,
          py::arg("hidden_states"),
          py::arg("w13"),
          py::arg("w2"),
          py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("global_num_experts"),
          py::arg("local_expert_start"),
          py::arg("local_num_experts"),
          R"doc(Ascend fused MoE path with expert-parallel routing.)doc");
}

} // namespace infinicore::ops
