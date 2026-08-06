#pragma once
#include "infinicore/ops/prepare_glm_w4a16_awq.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace infinicore {
inline void bind_prepare_glm_w4a16_awq(py::module &m) {
    m.def("prepare_glm_w4a16_awq_", &op::prepare_glm_w4a16_awq_,
          py::arg("qweight"), py::arg("qzeros"), py::arg("scales"),
          py::arg("checkpoint_weight"), py::arg("channel_scales"));
}
} // namespace infinicore
