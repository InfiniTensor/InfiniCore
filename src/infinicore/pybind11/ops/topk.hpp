#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/topk.hpp"

namespace py = pybind11;

namespace infinicore::ops {

//torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)

std::pair<Tensor, Tensor> py_topk(Tensor input, size_t k, py::object dim, bool largest, bool sorted){
      if(dim.is_none()){
            return op::topk(input, k, input->ndim() - 1,largest, sorted);
      } else if (py::isinstance<py::int_>(dim)){
            return op::topk(input, k, dim.cast<size_t>(),largest, sorted);
      } else {
            throw std::invalid_argument("invalid argument: dim");
      }
}

void py_topk_(Tensor values_output, Tensor indices_output, Tensor input, size_t k, py::object dim, bool largest, bool sorted){
      if(dim.is_none()){
            op::topk_(values_output, indices_output, input, k, input->ndim() - 1,largest, sorted);
      } else if (py::isinstance<py::int_>(dim)){
            op::sum_(values_output, indices_output, input, k, dim.cast<size_t>(),largest, sorted);
      } else {
            throw std::invalid_argument("invalid argument: dim");
      }
}

// todo 修改参数
inline void bind_topk(py::module &m) {
    m.def("topk",
      //     &op::topk,
          &py_topk,
          py::arg("input"),
          py::arg("k"),
          py::arg("dim"),
          py::arg("largest"),
          py::arg("sorted"),
          R"doc(topk of input tensor along the given dimensions.)doc");

    m.def("topk_",
      //     &op::topk_,
          &py_topk_,
          py::arg("values_output"),
          py::arg("indices_output"),
          py::arg("input"),
          py::arg("k"),
          py::arg("dim"),
          py::arg("largest"),
          py::arg("sorted"),
          R"doc(In-place tensor topk_.)doc");
}

} // namespace infinicore::ops
