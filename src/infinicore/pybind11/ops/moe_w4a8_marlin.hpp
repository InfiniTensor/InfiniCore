#pragma once

#include "infinicore/ops/moe_w4a8_marlin.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_moe_w4a8_marlin(py::module &m) {
    m.def("prepare_w4a8_marlin_weight_",
          &op::prepare_w4a8_marlin_weight_,
          py::arg("output"), py::arg("input"));
    m.def("moe_align_block_size_from_counts_",
          &op::moe_align_block_size_from_counts_,
          py::arg("padded_sorted_token_ids"), py::arg("expert_ids"),
          py::arg("num_tokens_post_pad"), py::arg("sorted_token_ids"),
          py::arg("tokens_per_expert"), py::arg("block_size"),
          py::arg("routing_topk"));
    m.def("moe_w4a8_marlin_",
          &op::moe_w4a8_marlin_,
          py::arg("output"), py::arg("input"), py::arg("marlin_weight"),
          py::arg("input_scale"), py::arg("weight_scale"),
          py::arg("topk_weights"), py::arg("padded_sorted_token_ids"), py::arg("expert_ids"),
          py::arg("num_tokens_post_pad"), py::arg("topk"),
          py::arg("routing_topk"));
}

} // namespace infinicore::ops
