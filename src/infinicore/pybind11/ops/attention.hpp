#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/attention.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_self_attention(Tensor query,
                         Tensor key,
                         Tensor value,
                         pybind11::object scale) {
    std::optional<float> scale_float = std::nullopt;
    if (!scale.is_none()) {
        scale_float = scale.cast<float>();
    }
    return op::self_attention(query, key, value, scale_float);
}

void py_self_attention_(Tensor out,
                        Tensor query,
                        Tensor key,
                        Tensor value,
                        pybind11::object scale) {
    std::optional<float> scale_float = std::nullopt;
    if (!scale.is_none()) {
        scale_float = scale.cast<float>();
    }
    op::self_attention_(out, query, key, value, scale_float);
}

inline void bind_attention(py::module &m) {
    m.def("attention",
          &op::attention,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("pos"),
          R"doc(Attention mechanism with KV caching.

Args:
    q: Query tensor
    k: Key tensor
    v: Value tensor
    k_cache: Key cache tensor
    v_cache: Value cache tensor
    pos: Current position in the sequence

Returns:
    Output tensor from attention computation
)doc");

    m.def("attention_",
          &op::Attention::execute,
          py::arg("out"),
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("pos"),
          R"doc(In-place attention mechanism with KV caching.

Args:
    out: Output tensor
    q: Query tensor
    k: Key tensor
    v: Value tensor  
    k_cache: Key cache tensor
    v_cache: Value cache tensor
    pos: Current position in the sequence
)doc");

    m.def("self_attention",
          &ops::py_self_attention,
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("scale") = py::none(),
          R"doc(Computes scaled dot product attention on query, key and value tensors)doc");

    m.def("self_attention_",
          &ops::py_self_attention_,
          py::arg("out"),
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("scale") = py::none(),
          R"doc(In-place, Computes scaled dot product attention on query, key and value tensors)doc");
}

} // namespace infinicore::ops
