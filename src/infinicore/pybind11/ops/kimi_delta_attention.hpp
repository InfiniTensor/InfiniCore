#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/ops/kimi_delta_attention.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_kimi_delta_attention(py::module &m) {
    m.def("kimi_delta_attention",
          &op::kimi_delta_attention,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("g"),
          py::arg("beta"),
          py::arg("A_log"),
          py::arg("dt_bias"),
          py::arg("initial_state"),
          py::arg("cu_seqlens") = std::nullopt,
          py::arg("initial_state_indices") = std::nullopt,
          py::arg("final_state_indices") = std::nullopt,
          py::arg("scale") = 1.0f,
          py::arg("lower_bound") = -5.0f,
          py::arg("use_qk_l2norm") = true,
          R"doc(Kimi Delta Attention out-of-place.)doc");

    m.def("kimi_delta_attention_",
          &op::kimi_delta_attention_,
          py::arg("out"),
          py::arg("initial_state"),
          py::arg("final_state"),
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("g"),
          py::arg("beta"),
          py::arg("A_log"),
          py::arg("dt_bias"),
          py::arg("cu_seqlens") = std::nullopt,
          py::arg("initial_state_indices") = std::nullopt,
          py::arg("final_state_indices") = std::nullopt,
          py::arg("scale") = 1.0f,
          py::arg("lower_bound") = -5.0f,
          py::arg("use_qk_l2norm") = true,
          R"doc(Kimi Delta Attention writing to provided output/state.)doc");
}

} // namespace infinicore::ops
