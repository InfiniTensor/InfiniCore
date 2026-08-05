#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/moe_fused_dense.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_moe_fused_dense(py::module &m) {
    m.def("moe_fused_dense",
          &op::moe_fused_dense,
          py::arg("hidden_states"),
          py::arg("w13"),
          py::arg("w2"),
          py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("sorted_token_ids"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_padded"),
          R"doc(Fused MoE dense path using aligned expert routing metadata.)doc");

    m.def("moe_fused_dense_",
          &op::moe_fused_dense_,
          py::arg("output"),
          py::arg("hidden_states"),
          py::arg("w13"),
          py::arg("w2"),
          py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("sorted_token_ids"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_padded"),
          R"doc(Fused MoE dense path writing to the provided output.)doc");
}

} // namespace infinicore::ops
