#pragma once

#include <pybind11/pybind11.h>

#include "op/matmul.hpp"
#include "op/rearrange.hpp"

namespace py = pybind11;

namespace infinicore::op {

inline void bind(py::module &m) {
    bind_matmul(m);
    bind_rearrange(m);
}

} // namespace infinicore::op
