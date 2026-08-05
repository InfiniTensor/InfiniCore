#pragma once

#include "infinicore/ops/select_last_token_hidden.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {
inline void bind_select_last_token_hidden(py::module &m) {
    m.def("select_last_token_hidden_",
          &op::select_last_token_hidden_,
          py::arg("output"),
          py::arg("hidden_states"),
          py::arg("input_offsets"));
}
} // namespace infinicore::ops
