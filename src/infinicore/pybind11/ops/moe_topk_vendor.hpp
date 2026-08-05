#pragma once
#include "infinicore/ops/moe_topk_vendor.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace infinicore::ops {
inline void bind_moe_topk_vendor(py::module &m) {
    m.def(
        "moe_topk_softmax_vendor_", [](Tensor topk_weights, Tensor topk_ids, Tensor token_expert_indices, Tensor gating_output, bool renormalize, py::object correction_bias) {
            Tensor bias;
            if (!correction_bias.is_none()) {
                bias = correction_bias.cast<Tensor>();
            }
            op::moe_topk_softmax_vendor_(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize, bias);
        },
        py::arg("topk_weights"), py::arg("topk_ids"), py::arg("token_expert_indices"), py::arg("gating_output"), py::arg("renormalize") = false, py::arg("correction_bias") = py::none(), R"doc(Iluvatar vendor MoE topk softmax with source_rows output.)doc");
    m.def(
        "moe_topk_sigmoid_vendor_", [](Tensor topk_weights, Tensor topk_ids, Tensor token_expert_indices, Tensor gating_output, bool renormalize, py::object correction_bias) {
            Tensor bias;
            if (!correction_bias.is_none()) {
                bias = correction_bias.cast<Tensor>();
            }
            op::moe_topk_sigmoid_vendor_(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize, bias);
        },
        py::arg("topk_weights"), py::arg("topk_ids"), py::arg("token_expert_indices"), py::arg("gating_output"), py::arg("renormalize") = false, py::arg("correction_bias") = py::none(), R"doc(Iluvatar vendor MoE topk sigmoid with source_rows output.)doc");
}
} // namespace infinicore::ops
