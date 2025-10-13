#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/matmul.hpp"
#include "ops/rearrange.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_matmul(m);
    bind_rearrange(m);
}

} // namespace infinicore::ops
