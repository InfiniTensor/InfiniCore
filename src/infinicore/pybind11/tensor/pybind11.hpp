#pragma once

#include "tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace infinicore::py::tensor {
inline void bind(pybind11::module &m) {
    pybind11::class_<infinicore::TensorSliceParams>(m, "TensorSliceParams")
        .def(pybind11::init<size_t, size_t, infinicore::Size>())
        .def_readwrite("dim", &infinicore::TensorSliceParams::dim)
        .def_readwrite("start", &infinicore::TensorSliceParams::start)
        .def_readwrite("len", &infinicore::TensorSliceParams::len);

    pybind11::class_<py::Tensor>(m, "Tensor")
        .def("copy_", &py::Tensor::copy_, pybind11::arg("src"))
        .def("to", &py::Tensor::to, pybind11::arg("device"))

        .def("shape", &py::Tensor::shape)
        .def("strides", &py::Tensor::strides)
        .def("dtype", &py::Tensor::dtype)
        .def("device", &py::Tensor::device)
        .def("is_contiguous", &py::Tensor::is_contiguous)
        .def("ndim", &py::Tensor::ndim)
        .def("numel", &py::Tensor::numel)
        .def("is_pinned", &py::Tensor::is_pinned)
        .def("info", &py::Tensor::info)

        .def("narrow", &py::Tensor::narrow, pybind11::arg("slices"))
        .def("permute", &py::Tensor::permute, pybind11::arg("order"))
        .def("view", &py::Tensor::view, pybind11::arg("new_shape"))
        .def("as_strided", &py::Tensor::as_strided,
             pybind11::arg("new_shape"), pybind11::arg("new_strides"));

    m.def("empty", &py::empty,
          pybind11::arg("shape"),
          pybind11::arg("dtype"),
          pybind11::arg("device"),
          pybind11::arg("pin_memory") = false);

    m.def("zeros", &py::zeros,
          pybind11::arg("shape"),
          pybind11::arg("dtype"),
          pybind11::arg("device"),
          pybind11::arg("pin_memory") = false);

    m.def("ones", &py::ones,
          pybind11::arg("shape"),
          pybind11::arg("dtype"),
          pybind11::arg("device"),
          pybind11::arg("pin_memory") = false);

    m.def("from_blob", &py::from_blob,
          pybind11::arg("raw_ptr"),
          pybind11::arg("shape"),
          pybind11::arg("dtype"),
          pybind11::arg("device"));
}
} // namespace infinicore::py::tensor
