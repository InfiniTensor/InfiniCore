#pragma once
#include "infinicore/ops/moe_argsort_bincount.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace infinicore::ops {
inline void bind_moe_argsort_bincount(py::module &m) {
    m.def("moe_argsort_bincount_with_inv_pos_", &op::moe_argsort_bincount_with_inv_pos_, py::arg("tokens_per_experts"), py::arg("sorted_indices"), py::arg("inv_pos"), py::arg("topk_ids"), py::arg("num_experts"));
}
} // namespace infinicore::ops
