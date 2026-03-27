#ifdef ENABLE_ATEN
#pragma once
#include "../context/context.hpp"
#include "../tensor.hpp"

#include <ATen/ATen.h>

#if defined(ENABLE_NVIDIA_API)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#elif defined(ENABLE_HYGON_API)
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#endif

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
    if (device.getType() == Device::Type::NVIDIA
        || device.getType() == Device::Type::HYGON) {
        return at::Device(at::kCUDA, device.getIndex());
    } else if (device.getType() == Device::Type::CPU) {
        return at::Device(at::kCPU);
    } else {
        throw std::runtime_error("Unsupported device type for ATen");
    }
}

at::Tensor to_aten_tensor(const infinicore::Tensor &t);

#if defined(ENABLE_HYGON_API)
using TorchStream = c10::hip::HIPStream;
using TorchStreamGuard = c10::hip::HIPStreamGuard;
TorchStream get_cuda_stream();
#elif defined(ENABLE_NVIDIA_API)
using TorchStream = c10::cuda::CUDAStream;
using TorchStreamGuard = c10::cuda::CUDAStreamGuard;
TorchStream get_cuda_stream();
#endif
} // namespace infinicore::adaptor

#endif // ENABLE_ATEN
