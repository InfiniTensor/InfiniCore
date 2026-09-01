#pragma once

#include "infinicore/ops/paged_attention_prefill.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_paged_attention_prefill(Tensor q,
                                  Tensor k_cache,
                                  Tensor v_cache,
                                  Tensor block_tables,
                                  Tensor history_lens,
                                  Tensor cu_seqlens_q,
                                  py::object alibi_slopes,
                                  float scale,
                                  py::object k_scale,
                                  py::object v_scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    std::optional<Tensor> k_scale_tensor = std::nullopt;
    if (!k_scale.is_none()) {
        k_scale_tensor = k_scale.cast<Tensor>();
    }
    std::optional<Tensor> v_scale_tensor = std::nullopt;
    if (!v_scale.is_none()) {
        v_scale_tensor = v_scale.cast<Tensor>();
    }
    return op::paged_attention_prefill(
        q, k_cache, v_cache, block_tables, history_lens, cu_seqlens_q, alibi_slopes_tensor, scale, k_scale_tensor, v_scale_tensor);
}

void py_paged_attention_prefill_(Tensor out,
                                 Tensor q,
                                 Tensor k_cache,
                                 Tensor v_cache,
                                 Tensor block_tables,
                                 Tensor history_lens,
                                 Tensor cu_seqlens_q,
                                 py::object alibi_slopes,
                                 float scale,
                                 py::object k_scale,
                                 py::object v_scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    std::optional<Tensor> k_scale_tensor = std::nullopt;
    if (!k_scale.is_none()) {
        k_scale_tensor = k_scale.cast<Tensor>();
    }
    std::optional<Tensor> v_scale_tensor = std::nullopt;
    if (!v_scale.is_none()) {
        v_scale_tensor = v_scale.cast<Tensor>();
    }
    op::paged_attention_prefill_(out, q, k_cache, v_cache, block_tables, history_lens, cu_seqlens_q, alibi_slopes_tensor, scale, k_scale_tensor, v_scale_tensor);
}

inline void bind_paged_attention_prefill(py::module &m) {
    m.def("paged_attention_prefill",
          &ops::py_paged_attention_prefill,
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("block_tables"),
          py::arg("history_lens"),
          py::arg("cu_seqlens_q"),
          py::arg("alibi_slopes") = py::none(),
          py::arg("scale") = 1.0,
          py::arg("k_scale") = py::none(),
          py::arg("v_scale") = py::none(),
          R"doc(Paged attention prefill for packed variable-length queries.)doc");

    m.def("paged_attention_prefill_",
          &ops::py_paged_attention_prefill_,
          py::arg("out"),
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("block_tables"),
          py::arg("history_lens"),
          py::arg("cu_seqlens_q"),
          py::arg("alibi_slopes") = py::none(),
          py::arg("scale") = 1.0,
          py::arg("k_scale") = py::none(),
          py::arg("v_scale") = py::none(),
          R"doc(In-place paged attention prefill for packed variable-length queries.)doc");
}

} // namespace infinicore::ops
