#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <infinicore.hpp>

namespace py = pybind11;

namespace infinicore {

PYBIND11_MODULE(infinicore, m) {
    py::class_<DataTypePy>(m, "dtype")
        .def("__repr__", DataTypePy::toString);

    m.attr("float32") = DataTypePy{DataType::F32};
    m.attr("float") = m.attr("float32");
    m.attr("float64") = DataTypePy{DataType::F64};
    m.attr("double") = m.attr("float64");
    m.attr("complex32") = DataTypePy{DataType::C32};
    m.attr("chalf") = m.attr("complex32");
    m.attr("complex64") = DataTypePy{DataType::C64};
    m.attr("cfloat") = m.attr("complex64");
    m.attr("complex128") = DataTypePy{DataType::C128};
    m.attr("cdouble") = m.attr("complex128");
    m.attr("float16") = DataTypePy{DataType::F16};
    m.attr("half") = m.attr("float16");
    m.attr("bfloat16") = DataTypePy{DataType::BF16};
    m.attr("uint8") = DataTypePy{DataType::U8};
    m.attr("int8") = DataTypePy{DataType::I8};
    m.attr("int16") = DataTypePy{DataType::I16};
    m.attr("short") = m.attr("int16");
    m.attr("int32") = DataTypePy{DataType::I32};
    m.attr("int") = m.attr("int32");
    m.attr("int64") = DataTypePy{DataType::I64};
    m.attr("long") = m.attr("int64");
    m.attr("bool") = DataTypePy{DataType::BOOL};

    py::class_<DevicePy>(m, "device")
        .def(py::init<const std::string &, Device::Index>(),
             py::arg("type"), py::arg("index"))
        .def_property_readonly("type", &DevicePy::getType)
        .def_property_readonly("index", &DevicePy::getIndex)
        .def("__repr__", &DevicePy::toRepresentation)
        .def("__str__", &DevicePy::toString);

    py::class_<Tensor>(m, "Tensor")
        .def_static("empty", &Tensor::empty,
                    py::arg("shape"), py::arg("dtype"), py::arg("device"))
        .def_static("zeros", &Tensor::zeros,
                    py::arg("shape"), py::arg("dtype"), py::arg("device"))
        .def_static("ones", &Tensor::ones,
                    py::arg("shape"), py::arg("dtype"), py::arg("device"))

        .def("shape", [](const Tensor &self) {
            return self->shape();
        })
        .def("dtype", [](const Tensor &self) {
            return self->dtype();
        })
        .def("device", [](const Tensor &self) {
            return self->device();
        });
}

} // namespace infinicore
