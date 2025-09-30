#pragma once

#include <pybind11/pybind11.h>

#include "op/matmul.hpp"

namespace py = pybind11;

namespace infinicore::op {

inline void bind(py::module &m) {
    bind_matmul(m);
}

} // namespace infinicore::op
