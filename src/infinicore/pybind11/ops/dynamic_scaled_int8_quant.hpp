#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/dynamic_scaled_int8_quant.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_dynamic_scaled_int8_quant(py::module &m) {
    m.def("dynamic_scaled_int8_quant_",
          &op::dynamic_scaled_int8_quant_,
          py::arg("output"),
          py::arg("input"),
          py::arg("input_scales"),
          R"doc(In-place per-token dynamic scaled int8 quantization via vllm_iluvatar on Iluvatar.)doc");

    m.def("dynamic_scaled_int8_quant",
          &op::dynamic_scaled_int8_quant,
          py::arg("input"),
          py::arg("input_scales"),
          R"doc(Per-token dynamic scaled int8 quantization via vllm_iluvatar on Iluvatar.)doc");
}

} // namespace infinicore::ops
