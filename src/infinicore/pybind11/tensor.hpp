#pragma once

#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore.hpp"

namespace py = pybind11;

namespace infinicore::tensor {

inline void bind(py::module &m) {
    py::class_<Tensor>(m, "Tensor")
        .def_property_readonly("shape", [](const Tensor &tensor) { return tensor->shape(); })
        .def_property_readonly("strides", [](const Tensor &tensor) { return tensor->strides(); })
        .def_property_readonly("ndim", [](const Tensor &tensor) { return tensor->ndim(); })
        .def_property_readonly("dtype", [](const Tensor &tensor) { return tensor->dtype(); })
        .def_property_readonly("device", [](const Tensor &tensor) { return tensor->device(); })
        .def("data_ptr", [](const Tensor &tensor) { return reinterpret_cast<std::uintptr_t>(tensor->data()); })
        .def("size", [](const Tensor &tensor, std::size_t dim) { return tensor->size(dim); })
        .def("stride", [](const Tensor &tensor, std::size_t dim) { return tensor->stride(dim); })
        .def("numel", [](const Tensor &tensor) { return tensor->numel(); })
        .def("is_contiguous", [](const Tensor &tensor) { return tensor->is_contiguous(); })
        .def("is_pinned", [](const Tensor &tensor) { return tensor->is_pinned(); })
        .def("info", [](const Tensor &tensor) { return tensor->info(); })

        .def("debug", [](const Tensor &tensor) { return tensor->debug(); })
        .def("debug", [](const Tensor &tensor, const std::string &filename) { return tensor->debug(filename); })

        .def("copy_", [](Tensor &tensor, const Tensor &other) { tensor->copy_from(other); })
        .def("to", [](const Tensor &tensor, const Device &device) { return tensor->to(device); })
        .def("contiguous", [](const Tensor &tensor) { return tensor->contiguous(); })

        .def("as_strided", [](const Tensor &tensor, const Shape &shape, const Strides &strides) { return tensor->as_strided(shape, strides); })
        .def("narrow", [](const Tensor &tensor, std::size_t dim, std::size_t start, std::size_t length) { return tensor->narrow({{dim, start, length}}); })
        .def("permute", [](const Tensor &tensor, const Shape &dims) { return tensor->permute(dims); })
        .def("view", [](const Tensor &tensor, const Shape &shape) { return tensor->view(shape); })
        .def("unsqueeze", [](const Tensor &tensor, std::size_t dim) { return tensor->unsqueeze(dim); })
        .def("squeeze", [](const Tensor &tensor, std::size_t dim) { return tensor->squeeze(dim); })

        // Fast in-place scalar writes for tiny CPU metadata tensors (decode loops, etc.).
        .def(
            "write_i32",
            [](Tensor &tensor, std::size_t linear_index, std::int32_t value) {
                if (tensor->dtype() != DataType::I32) {
                    throw py::value_error("write_i32: dtype must be I32");
                }
                if (!tensor->is_contiguous()) {
                    throw py::value_error("write_i32: tensor must be contiguous");
                }
                if (linear_index >= static_cast<std::size_t>(tensor->numel())) {
                    throw py::index_error("write_i32: linear_index out of range");
                }
                auto *base = reinterpret_cast<std::int32_t *>(tensor->data());
                base[linear_index] = value;
            },
            py::arg("linear_index"),
            py::arg("value"))
        .def(
            "write_i64",
            [](Tensor &tensor, std::size_t linear_index, std::int64_t value) {
                if (tensor->dtype() != DataType::I64) {
                    throw py::value_error("write_i64: dtype must be I64");
                }
                if (!tensor->is_contiguous()) {
                    throw py::value_error("write_i64: tensor must be contiguous");
                }
                if (linear_index >= static_cast<std::size_t>(tensor->numel())) {
                    throw py::index_error("write_i64: linear_index out of range");
                }
                auto *base = reinterpret_cast<std::int64_t *>(tensor->data());
                base[linear_index] = value;
            },
            py::arg("linear_index"),
            py::arg("value"));

    m.def("empty", &Tensor::empty,
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("pin_memory") = false);
    m.def("strided_empty", &Tensor::strided_empty,
          py::arg("shape"),
          py::arg("strides"),
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

    m.def(
        "from_blob", [](uintptr_t raw_ptr, Shape &shape, const DataType &dtype, const Device &device) {
            return Tensor{infinicore::Tensor::from_blob(reinterpret_cast<void *>(raw_ptr), shape, dtype, device)};
        },
        pybind11::arg("raw_ptr"), pybind11::arg("shape"), pybind11::arg("dtype"), pybind11::arg("device"));

    m.def(
        "strided_from_blob", [](uintptr_t raw_ptr, Shape &shape, Strides &strides, const DataType &dtype, const Device &device) {
            return Tensor{infinicore::Tensor::strided_from_blob(reinterpret_cast<void *>(raw_ptr), shape, strides, dtype, device)};
        },
        pybind11::arg("raw_ptr"), pybind11::arg("shape"), pybind11::arg("strides"), pybind11::arg("dtype"), pybind11::arg("device"));
}

} // namespace infinicore::tensor
