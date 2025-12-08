#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/sum.hpp"

namespace py = pybind11;

namespace infinicore::ops {


Tensor py_sum(Tensor input, py::object dim, bool keepdim){
      if(dim.is_none()){
            std::vector<size_t> dim_vec;
            for(int i = 0; i < input->shape().size(); i++){
                dim_vec.push_back(i);
            }
            return op::sum(input, dim_vec, keepdim);
      } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)){
            return op::sum(input, dim.cast<std::vector<size_t>>(), keepdim);
      } else if (py::isinstance<py::int_>(dim)){
            return op::sum(input, std::vector<size_t>(1, dim.cast<size_t>()), keepdim);
      } else {
            throw std::invalid_argument("dim must be a tuple or an integer");
      }
}

Tensor py_sum_(Tensor output, Tensor input, py::object dim, bool keepdim){
      if(dim.is_none()){
            std::vector<size_t> dim_vec;
            for(int i = 0; i < input->shape().size(); i++){
                dim_vec.push_back(i);
            }
            return op::sum(output,input, dim_vec, keepdim);
      } else if (py::isinstance<py::tuple>(dim) || py::isinstance<py::list>(dim)){
            return op::sum(output, input, dim.cast<std::vector<size_t>>(), keepdim);
      } else if (py::isinstance<py::int_>(dim)){
            return op::sum(output, input, std::vector<size_t>(1, dim.cast<size_t>()), keepdim);
      } else {
            throw std::invalid_argument("dim must be a tuple or an integer");
      }
}

// todo 修改参数
inline void bind_sum(py::module &m) {
    m.def("sum",
      //     &op::sum,
          &py_sum,
          py::arg("input"),
          py::arg("dim"), // = py::none()
          py::arg("keepdim"), // = false  todo 搞清楚这两个参数有没有默认值的区别
          R"doc(Sum of input tensor along the given dimensions.)doc");

    m.def("sum_",
      //     &op::sum_,
          &py_sum_,
          py::arg("output"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("keepdim"),
          R"doc(In-place tensor sum.)doc");
}

} // namespace infinicore::ops
