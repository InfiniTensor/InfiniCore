#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::tensor {

inline void bind(py::module &m) {
    py::class_<Tensor>(m, "Tensor")
        .def_property_readonly("shape", [](const Tensor &tensor) { return tensor->shape(); })
        .def_property_readonly("strides", [](const Tensor &tensor) { return tensor->device(); })
        .def_property_readonly("ndim", [](const Tensor &tensor) { return tensor->ndim(); })

        .def("data_ptr", [](const Tensor &tensor) { return tensor->data(); })
        .def("size", [](const Tensor &tensor, std::size_t dim) { return tensor->size(dim); })
        .def("stride", [](const Tensor &tensor, std::size_t dim) { return tensor->stride(dim); })
        .def("numel", [](const Tensor &tensor) { return tensor->numel(); })

        .def("is_contiguous", [](const Tensor &tensor) { return tensor->is_contiguous(); })
        .def("is_pinned", [](const Tensor &tensor) { return tensor->is_pinned(); })
        .def("info", [](const Tensor &tensor) { return tensor->info(); })

        .def("copy_", [](Tensor &tensor, const Tensor &other) { tensor->copy_from(other); })
        .def("to", [](const Tensor &tensor, const Device &device) { return tensor->to(device); })
        .def("as_strided", [](const Tensor &tensor, const Shape &shape, const Strides &strides) { return tensor->as_strided(shape, strides); })
        .def("contiguous", [](const Tensor &tensor) { return tensor->contiguous(); })

        .def("permute", [](const Tensor &tensor, const Shape &dims) { return tensor->permute(dims); })
        .def("view", [](const Tensor &tensor, const Shape &shape) { return tensor->view(shape); });

    m.def("empty", &Tensor::empty,
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("pin_memory") = false);
    m.def("zeros", &Tensor::zeros,
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("pin_memory") = false);
    m.def("ones", &Tensor::ones,
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("pin_memory") = false);

    m.def("from_blob", [](uintptr_t raw_ptr, Shape &shape, const DataType &dtype, const Device &device) { return Tensor{infinicore::Tensor::from_blob(reinterpret_cast<void *>(raw_ptr), shape, dtype, device)}; }, pybind11::arg("raw_ptr"), pybind11::arg("shape"), pybind11::arg("dtype"), pybind11::arg("device"));
}

} // namespace infinicore::tensor
