#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device/pybind11.hpp"
#include "dtype/pybind11.hpp"
#include "op/pybind11.hpp"
#include "tensor/pybind11.hpp"

namespace infinicore {

PYBIND11_MODULE(infinicore, m) {
    py::device::bind(m);
    py::dtype::bind(m);
    py::tensor::bind(m);
    py::op::bind(m);
}

} // namespace infinicore
