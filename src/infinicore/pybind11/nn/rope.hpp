#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/nn/rope.hpp"

namespace py = pybind11;

namespace infinicore::pybind11_nn {

inline void bind_rope(py::module &m) {
    py::enum_<infinicore::nn::RoPE::FreqGen>(m, "RoPEFreqGen")
        .value("GPT_J", infinicore::nn::RoPE::FreqGen::GPT_J)
        .value("GPT_NEOX", infinicore::nn::RoPE::FreqGen::GPT_NEOX);

    py::enum_<infinicore::nn::RoPE::Algo>(m, "RoPEAlgo")
        .value("GPT_J", infinicore::nn::RoPE::Algo::GPT_J)
        .value("GPT_NEOX", infinicore::nn::RoPE::Algo::GPT_NEOX);

    py::class_<infinicore::nn::RoPE>(m, "RoPE")
        .def(py::init<size_t, size_t, double, infinicore::nn::RoPE::FreqGen, infinicore::nn::RoPE::Algo, const DataType &, const Device &>(),
             py::arg("head_dim"),
             py::arg("max_seq_len"),
             py::arg("theta") = 10000.0,
             py::arg("freq_gen") = infinicore::nn::RoPE::FreqGen::GPT_J,
             py::arg("algo") = infinicore::nn::RoPE::Algo::GPT_J,
             py::arg("dtype") = DataType::F32,
             py::arg("device") = Device())
        .def(
            "forward", [](infinicore::nn::RoPE &self, py::object x_obj, py::object pos_obj) {
                // Unwrap Python Tensor wrappers if needed
                infinicore::Tensor x = py::hasattr(x_obj, "_underlying")
                                         ? x_obj.attr("_underlying").cast<infinicore::Tensor>()
                                         : x_obj.cast<infinicore::Tensor>();

                infinicore::Tensor pos = py::hasattr(pos_obj, "_underlying")
                                           ? pos_obj.attr("_underlying").cast<infinicore::Tensor>()
                                           : pos_obj.cast<infinicore::Tensor>();

                return self.forward(x, pos);
            },
            py::arg("x"), py::arg("pos"))
        .def("head_dim", &infinicore::nn::RoPE::head_dim)
        .def("max_seq_len", &infinicore::nn::RoPE::max_seq_len)
        .def("theta", &infinicore::nn::RoPE::theta)
        .def("freq_gen", &infinicore::nn::RoPE::freq_gen)
        .def("algo", &infinicore::nn::RoPE::algo)
        .def("dtype", &infinicore::nn::RoPE::dtype)
        .def("extra_repr", &infinicore::nn::RoPE::extra_repr);
}

} // namespace infinicore::pybind11_nn
