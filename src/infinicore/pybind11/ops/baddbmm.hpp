#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/dtype.hpp"
#include "infinicore/ops/baddbmm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

namespace {
// Helper function to create a scalar tensor with the correct dtype
Tensor create_scalar_tensor(float value, DataType dtype, Device device) {
    // Create buffer for the target dtype
    alignas(8) char buffer[8];
    convertFloat(static_cast<double>(value), dtype, buffer);    
    
    // Create CPU tensor with correct dtype
    Tensor cpu_tensor = Tensor::from_blob(buffer, {}, dtype, Device(Device::Type::CPU));
    
    // Create device tensor and copy
    Tensor device_tensor = Tensor::empty({}, dtype, device);
    device_tensor->copy_from(cpu_tensor);
    
    return device_tensor;
}
}

Tensor py_baddbmm(Tensor input, Tensor batch1, Tensor batch2, float beta = 1.0f, float alpha = 1.0f) {
    if(beta != 1.0f ||  alpha != 1.0f) {
        DataType dtype = batch1->dtype();
        Device device = input->device();
        
        Tensor beta_tensor = create_scalar_tensor(beta, dtype, device);
        Tensor alpha_tensor = create_scalar_tensor(alpha, dtype, device);
        
        return op::baddbmm(input, batch1, batch2, beta_tensor, alpha_tensor);
    }
    return op::baddbmm(input, batch1, batch2);
}

void py_baddbmm_(Tensor out, Tensor input, Tensor batch1, Tensor batch2, float beta = 1.0f, float alpha = 1.0f) {
    if(beta != 1.0f ||  alpha != 1.0f) {
        DataType dtype = batch1->dtype();
        Device device = input->device();
        
        Tensor beta_tensor = create_scalar_tensor(beta, dtype, device);
        Tensor alpha_tensor = create_scalar_tensor(alpha, dtype, device);

        op::baddbmm_(out, input, batch1, batch2, beta_tensor, alpha_tensor);
        return;
    }
    op::baddbmm_(out, input, batch1, batch2);
}

inline void bind_baddbmm(py::module &m) {
    m.def("baddbmm",
          &py_baddbmm,
          py::arg("input"),
          py::arg("batch1"),
          py::arg("batch2"),
          py::arg("beta") = 1.0f,
          py::arg("alpha") = 1.0f,
          R"doc(Batched matrix-matrix product with addition.
Args:
    input: Input tensor
    batch1: First batch of matrices
    batch2: Second batch of matrices
    beta: Scaling factor for input tensor
    alpha: Scaling factor for the product of batch1 and batch2
Returns:
    Output tensor after baddbmm operation
)doc");
        m.def("baddbmm_",
                &py_baddbmm_,
                py::arg("out"),
                py::arg("input"),
                py::arg("batch1"),
                py::arg("batch2"),
                py::arg("beta") = 1.0f,
                py::arg("alpha") = 1.0f,
                R"doc(In-place batched matrix-matrix product with addition.
Args:
    out: Output tensor
    input: Input tensor
    batch1: First batch of matrices
    batch2: Second batch of matrices
    beta: Scaling factor for input tensor
    alpha: Scaling factor for the product of batch1 and batch2
)doc");
}

} // namespace infinicore::ops