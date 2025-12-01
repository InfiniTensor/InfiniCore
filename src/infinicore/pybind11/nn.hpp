#pragma once

#include <pybind11/pybind11.h>

#include "nn/rope.hpp"

namespace py = pybind11;

namespace infinicore::pybind11_nn {

inline void bind(py::module &m) {
    bind_rope(m);
}

} // namespace infinicore::pybind11_nn
