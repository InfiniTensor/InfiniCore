#pragma once

#include <pybind11/pybind11.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::context {

inline void bind(py::module &m) {
    m.def("get_device", &getDevice);
    m.def("get_device_count", &getDeviceCount);
}

} // namespace infinicore::context
