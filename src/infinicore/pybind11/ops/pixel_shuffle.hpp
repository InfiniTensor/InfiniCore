#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/pixel_shuffle.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_pixel_shuffle(py::module &m) {
    m.def("pixel_shuffle", &infinicore::op::pixel_shuffle);
}

} // namespace infinicore::ops

