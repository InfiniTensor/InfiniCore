#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/paged_attention_v1.hpp"

namespace py = pybind11;

namespace infinicore::ops {

void py_paged_attention_v1(
    Tensor out,         // [num_seqs, num_heads, head_size]
    Tensor query,       // [num_seqs, num_heads, head_size]
    Tensor key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    Tensor value_cache, // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,
    double scale,
    Tensor block_tables, // [num_seqs, max_num_blocks_per_seq]
    Tensor seq_lens,     // [num_seqs]
    int64_t block_size,
    int64_t max_seq_len,
    pybind11::object alibi_slopes,
    const std::string &kv_cache_dtype,
    Tensor k_scale,
    Tensor v_scale,
    int64_t tp_rank,
    int64_t blocksparse_local_blocks,
    int64_t blocksparse_vert_stride,
    int64_t blocksparse_block_size,
    int64_t blocksparse_head_sliding_step) {

    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    op::paged_attention_v1(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
        alibi_slopes_tensor, kv_cache_dtype, k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step);
}

inline void bind_paged_attention_v1(py::module &m) {

    m.def("paged_attention_v1",
          &ops::py_paged_attention_v1,
          py::arg("out"),
          py::arg("query"),
          py::arg("key_cache"),
          py::arg("value_cache"),
          py::arg("num_kv_heads"),
          py::arg("scale"),
          py::arg("block_tables"),
          py::arg("seq_lens"),
          py::arg("block_size"),
          py::arg("max_seq_len"),
          py::arg("alibi_slopes"),
          py::arg("kv_cache_dtype"),
          py::arg("k_scale"),
          py::arg("v_scale"),
          py::arg("tp_rank"),
          py::arg("blocksparse_local_blocks"),
          py::arg("blocksparse_vert_stride"),
          py::arg("blocksparse_block_size"),
          py::arg("blocksparse_head_sliding_step"),
          R"doc(paged attention v1 of query and key cache tensors.)doc");
}

} // namespace infinicore::ops
