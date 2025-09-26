#pragma once

#include "device.hpp"

#include <pybind11/pybind11.h>

namespace infinicore::py::device {
inline void bind(pybind11::module &m) {
    pybind11::class_<py::Device>(m, "device")
        .def(pybind11::init<const std::string &, Device::Index>(),
             pybind11::arg("type"), pybind11::arg("index"))
        .def_property_readonly("type", &py::Device::getType)
        .def_property_readonly("index", &py::Device::getIndex)
        .def("__repr__", &py::Device::toRepresentation)
        .def("__str__", &py::Device::toString);
}
} // namespace infinicore::py::device
