#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <infinicore.hpp>

namespace py = pybind11;

namespace infinicore {

PYBIND11_MODULE(infinicore, m) {
    py::enum_<DataType>(m, "dtype")
        .value("bfloat16", DataType::bfloat16)
        .value("float16", DataType::float16)
        .value("float32", DataType::float32)
        .value("float64", DataType::float64)
        .value("int32", DataType::int32)
        .value("int64", DataType::int64)
        .value("uint8", DataType::uint8)
        .export_values();

    py::class_<Device>(m, "Device")
        .def(py::init<const Device::Type &, const Device::Index &>(),
             py::arg("type"), py::arg("index") = 0)
        .def_property_readonly("type", &Device::get_type)
        .def_property_readonly("index", &Device::get_index)
        .def("__repr__", static_cast<std::string (Device::*)() const>(&Device::to_string));

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const Tensor::Shape &, const DataType &, const Device &>(),
             py::arg("shape"), py::arg("dtype") = DataType::float32, py::arg("device") = Device{Device::Type::cpu})
        .def_property_readonly("shape", &Tensor::get_shape)
        .def_property_readonly("dtype", &Tensor::get_dtype)
        .def_property_readonly("device", &Tensor::get_device);
}

} // namespace infinicore
