#pragma once

#include "infinicore/ops/dsa.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore {
inline void bind_dsa(py::module &m) {
    m.def("fused_deepseek_v2_indexer_postprocess_",
          &op::fused_deepseek_v2_indexer_postprocess_);
    m.def("indexer_k_cache_", &op::indexer_k_cache_);
    m.def("compute_block_sparse_mqa_logits_",
          &op::compute_block_sparse_mqa_logits_);
    m.def("select_prefill_topk_block_indices_",
          &op::select_prefill_topk_block_indices_);
    m.def("select_decode_topk_block_indices_",
          &op::select_decode_topk_block_indices_);
    m.def("map_prefill_request_block_indices_",
          &op::map_prefill_request_block_indices_,
          py::arg("output"),
          py::arg("req_id"),
          py::arg("block_table"),
          py::arg("token_indices"),
          py::arg("block_size"),
          py::arg("has_prefill_workspace") = false,
          py::arg("prefill_workspace_request_ids") = std::nullopt,
          py::arg("prefill_workspace_starts") = std::nullopt);
    m.def("map_decode_request_block_indices_",
          &op::map_decode_request_block_indices_);
    m.def("topk_indices_context_lens_",
          &op::topk_indices_context_lens_);
    m.def("sparse_flash_mla_",
          &op::sparse_flash_mla_,
          py::arg("output"),
          py::arg("query"),
          py::arg("kv_cache"),
          py::arg("indices"),
          py::arg("topk_lens"),
          py::arg("scale"),
          py::arg("attn_sink") = std::nullopt);
}
} // namespace infinicore
