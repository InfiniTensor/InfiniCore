#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/paged_caching.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void py_paged_caching_(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping, py::object k_scale, py::object v_scale) {
    std::optional<Tensor> k_scale_tensor = std::nullopt;
    if (!k_scale.is_none()) {
        k_scale_tensor = k_scale.cast<Tensor>();
    }
    std::optional<Tensor> v_scale_tensor = std::nullopt;
    if (!v_scale.is_none()) {
        v_scale_tensor = v_scale.cast<Tensor>();
    }
    op::paged_caching_(k_cache, v_cache, k, v, slot_mapping, k_scale_tensor, v_scale_tensor);
}

inline void bind_paged_caching(py::module &m) {
    m.def("paged_caching_",
          &ops::py_paged_caching_,
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("k"),
          py::arg("v"),
          py::arg("slot_mapping"),
          py::arg("k_scale") = py::none(),
          py::arg("v_scale") = py::none(),
          R"doc(Paged caching of key and value tensors.)doc");
}

} // namespace infinicore::ops
