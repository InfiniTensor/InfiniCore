#pragma once
#include "../tensor.hpp"

#include <ATen/ATen.h>

namespace infinicore::adaptor {
inline at::ScalarType to_at_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::F32:
        return at::kFloat;
    case DataType::F16:
        return at::kHalf;
    case DataType::BF16:
        return at::kBFloat16;
    case DataType::I32:
        return at::kInt;
    case DataType::I64:
        return at::kLong;
    default:
        throw std::runtime_error("Unsupported dtype for ATen");
    }
}

inline at::Device to_at_device(const Device &device) {
    if (device.getType() == Device::Type::NVIDIA) {
        return at::Device(at::kCUDA, device.getIndex());
    } else if (device.getType() == Device::Type::CPU) {
        return at::Device(at::kCPU);
    } else {
        throw std::runtime_error("Unsupported device type for ATen");
    }
}

at::Tensor to_aten_tensor(const infinicore::Tensor &t);
} // namespace infinicore::adaptor