#pragma once

#include <pybind11/pybind11.h>

#include "matmul/matmul.hpp"
// #include "add/add.hpp"
// #include "conv/conv.hpp"

namespace infinicore::py::op {

inline void bind(pybind11::module &m) {
    bind_matmul(m);
    // bind_add(m);
    // bind_conv(m);
}

} // namespace infinicore::py::op
