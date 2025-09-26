#pragma once

#include "dtype.hpp"

#include <pybind11/pybind11.h>

namespace infinicore::py::dtype {
inline void bind(pybind11::module &m) {
    pybind11::class_<py::DataType>(m, "dtype")
        .def("__repr__", py::DataType::toString);

    m.attr("float32") = py::DataType{infinicore::DataType::F32};
    m.attr("float") = m.attr("float32");
    m.attr("float64") = py::DataType{infinicore::DataType::F64};
    m.attr("double") = m.attr("float64");
    m.attr("complex32") = py::DataType{infinicore::DataType::C32};
    m.attr("chalf") = m.attr("complex32");
    m.attr("complex64") = py::DataType{infinicore::DataType::C64};
    m.attr("cfloat") = m.attr("complex64");
    m.attr("complex128") = py::DataType{infinicore::DataType::C128};
    m.attr("cdouble") = m.attr("complex128");
    m.attr("float16") = py::DataType{infinicore::DataType::F16};
    m.attr("half") = m.attr("float16");
    m.attr("bfloat16") = py::DataType{infinicore::DataType::BF16};
    m.attr("uint8") = py::DataType{infinicore::DataType::U8};
    m.attr("int8") = py::DataType{infinicore::DataType::I8};
    m.attr("int16") = py::DataType{infinicore::DataType::I16};
    m.attr("short") = m.attr("int16");
    m.attr("int32") = py::DataType{infinicore::DataType::I32};
    m.attr("int") = m.attr("int32");
    m.attr("int64") = py::DataType{infinicore::DataType::I64};
    m.attr("long") = m.attr("int64");
    m.attr("bool") = py::DataType{infinicore::DataType::BOOL};
}
} // namespace infinicore::py::dtype
