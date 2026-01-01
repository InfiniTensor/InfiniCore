#include "../tensor.hpp"
#include <pybind11/pybind11.h>
#include "infinicore/ops/float_power.hpp"

namespace py = pybind11;

namespace infinicore::ops {

using infinicore::Tensor;
using infinicore::op::float_power;
using infinicore::op::float_power_;

// 定义一个通用的解包函数，专门处理你提供的 Python Tensor 类
inline Tensor unwrap(py::handle obj) {
    // 1. 尝试直接从 Pybind11 注册的 C++ 类型转换 (预防万一直接传了 _underlying)
    try {
        return obj.cast<Tensor>();
    } catch (...) {}

    // 2. 穿透 Python 包装类提取 _underlying (这是核心)
    if (py::hasattr(obj, "_underlying")) {
        return obj.attr("_underlying").cast<Tensor>();
    }

    throw py::type_error("Expected infinicore.Tensor, but got " + py::repr(obj.get_type()).cast<std::string>());
}

void bind_float_power(py::module &m) {

    // --- Out-of-place: float_power(input, exponent) ---
    m.def("float_power", [](py::object input_obj, py::object exp_obj) -> Tensor {
        Tensor input = unwrap(input_obj);
        
        // 处理标量指数的情况 (float 或 int)
        if (py::isinstance<py::float_>(exp_obj) || py::isinstance<py::int_>(exp_obj)) {
            return float_power(input, exp_obj.cast<double>());
        }
        
        // 处理张量指数的情况
        Tensor exponent = unwrap(exp_obj);
        return float_power(input, exponent);
    }, py::arg("input"), py::arg("exponent"));

    // --- In-place: float_power_(out, input, exponent) ---
    m.def("float_power_", [](py::object out_obj, py::object input_obj, py::object exp_obj) {
        Tensor out = unwrap(out_obj);
        Tensor input = unwrap(input_obj);
        
        if (py::isinstance<py::float_>(exp_obj) || py::isinstance<py::int_>(exp_obj)) {
            float_power_(out, input, exp_obj.cast<double>());
        } else {
            Tensor exponent = unwrap(exp_obj);
            float_power_(out, input, exponent);
        }
    }, py::arg("out"), py::arg("input"), py::arg("exponent"));
}

} // namespace infinicore::ops