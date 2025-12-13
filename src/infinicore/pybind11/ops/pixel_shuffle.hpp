#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/pixel_shuffle.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_pixel_shuffle(py::module &m) {
    m.def("pixel_shuffle", &infinicore::op::pixel_shuffle, 
          py::arg("input"), py::arg("upscale_factor"),
          "Rearranges elements in a tensor of shape (*, C*r^2, H, W) to a tensor of shape (*, C, H*r, W*r).");
}

} // namespace infinicore::ops

