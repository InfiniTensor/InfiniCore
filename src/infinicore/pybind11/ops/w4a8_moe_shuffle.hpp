#pragma once

#include "infinicore/ops/w4a8_moe_shuffle.hpp"
#include <pybind11/pybind11.h>

namespace infinicore::ops {

inline void bind_w4a8_moe_shuffle(pybind11::module &m) {
    m.def("w4a8_moe_shuffle", &op::w4a8_moe_shuffle,
          pybind11::arg("input"));
    m.def("w4a8_moe_shuffle_", &op::w4a8_moe_shuffle_,
          pybind11::arg("output"), pybind11::arg("input"));
}

} // namespace infinicore::ops
