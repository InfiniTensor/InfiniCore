#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <infinicore.hpp>

namespace py = pybind11;

namespace infinicore {

PYBIND11_MODULE(infinicore, m) {
    py::enum_<DataType>(m, "dtype")
        .value("bfloat16", DataType::BFLOAT16)
        .value("float16", DataType::FLOAT16)
        .value("float32", DataType::FLOAT32)
        .value("float64", DataType::FLOAT64)
        .value("int32", DataType::INT32)
        .value("int64", DataType::INT64)
        .value("uint8", DataType::UINT8)
        .export_values();

    py::class_<Device>(m, "Device")
        .def(py::init<const Device::Type &, const Device::Index &>(),
             py::arg("type"), py::arg("index") = 0)
        .def_property_readonly("type", &Device::getType)
        .def_property_readonly("index", &Device::getIndex)
        .def("__repr__", static_cast<std::string (Device::*)() const>(&Device::toString));

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const Tensor::Shape &, const DataType &, const Device &>(),
             py::arg("shape"), py::arg("dtype") = DataType::FLOAT32, py::arg("device") = Device{Device::Type::CPU})
        .def_property_readonly("shape", &Tensor::getShape)
        .def_property_readonly("dtype", &Tensor::getDtype)
        .def_property_readonly("device", &Tensor::getDevice);
}

} // namespace infinicore
