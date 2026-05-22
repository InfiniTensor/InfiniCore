#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/infllmv2_attention.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline Tensor py_infllmv2_attention_varlen(Tensor q,
                                           Tensor k,
                                           Tensor v,
                                           Tensor cu_seqlens_q,
                                           Tensor cu_seqlens_k,
                                           int max_seqlen_q,
                                           int max_seqlen_k,
                                           float scale,
                                           bool causal,
                                           int window_size_left,
                                           int window_size_right) {
    return op::infllmv2_attention_varlen(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        scale, causal,
        window_size_left, window_size_right);
}

inline Tensor py_infllmv2_attention_kvcache(Tensor q,
                                            Tensor k_cache,
                                            Tensor v_cache,
                                            Tensor cache_lens,
                                            float scale,
                                            bool causal,
                                            int window_size_left,
                                            int window_size_right) {
    return op::infllmv2_attention_kvcache(
        q, k_cache, v_cache,
        cache_lens,
        scale, causal,
        window_size_left, window_size_right);
}

inline void bind_infllmv2_attention(py::module &m) {
    m.def(
        "infllmv2_attention_varlen",
        &ops::py_infllmv2_attention_varlen,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"),
        py::arg("max_seqlen_q"),
        py::arg("max_seqlen_k"),
        py::arg("scale"),
        py::arg("causal"),
        py::arg("window_size_left") = -1,
        py::arg("window_size_right") = -1,
        R"doc(InfLLM-V2 varlen attention. q,k,v unpadded; cu_seqlens_q/k [batch+1]. Returns [total_q, nheads, head_dim].)doc");

    m.def(
        "infllmv2_attention_kvcache",
        &ops::py_infllmv2_attention_kvcache,
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("cache_lens"),
        py::arg("scale"),
        py::arg("causal"),
        py::arg("window_size_left") = -1,
        py::arg("window_size_right") = -1,
        R"doc(InfLLM-V2 KV-cache (decode) attention. Returns [batch, seqlen_q, nheads, head_dim].)doc");
}

} // namespace infinicore::ops
