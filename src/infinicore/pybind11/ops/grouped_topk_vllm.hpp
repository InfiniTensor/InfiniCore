#pragma once
#include "infinicore/ops/grouped_topk_vllm.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace infinicore::ops {
inline void bind_grouped_topk_vllm(py::module &m) {
    m.def(
        "grouped_topk_vllm_", [](Tensor topk_weights, Tensor topk_ids, Tensor scores, int64_t num_expert_group, int64_t topk_group, bool renormalize, float routed_scaling_factor, py::object bias, const std::string &scoring_func) {
            Tensor b;
            if (!bias.is_none()) {
                b = bias.cast<Tensor>();
            }
            op::grouped_topk_vllm_(topk_weights, topk_ids, scores, num_expert_group, topk_group, renormalize, routed_scaling_factor, b, scoring_func);
        },
        py::arg("topk_weights"), py::arg("topk_ids"), py::arg("scores"), py::arg("num_expert_group"), py::arg("topk_group"), py::arg("renormalize"), py::arg("routed_scaling_factor"), py::arg("bias") = py::none(), py::arg("scoring_func") = "softmax", R"doc(vLLM-style grouped_topk routing.)doc");
}
} // namespace infinicore::ops
