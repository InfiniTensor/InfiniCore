#ifdef ENABLE_ATEN
#pragma once
#include "../context/context.hpp"
#include "../tensor.hpp"

#include <ATen/ATen.h>

#if defined(ENABLE_HYGON_API)
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#elif defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#if defined(ENABLE_MOORE_API)
#include <c10/macros/Macros.h>
#include <c10/musa/MUSAMacros.h>
#include <c10/musa/MUSAStream.h>
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
    // PyTorch ATen only exposes standard device types (e.g. kCPU/kCUDA).
    // Treat MetaX/QY devices as CUDA devices for ATen tensor interoperability.
    if (device.getType() == Device::Type::NVIDIA || device.getType() == Device::Type::METAX || device.getType() == Device::Type::MARS || device.getType() == Device::Type::QY || device.getType() == Device::Type::HYGON) {
        return at::Device(at::kCUDA, device.getIndex());
    } else if (device.getType() == Device::Type::CPU) {
        return at::Device(at::kCPU);
    }
#if defined(ENABLE_MOORE_API)
    else if (device.getType() == Device::Type::MOORE) {
        return at::Device(at::DeviceType::PrivateUse1, device.getIndex());
    }
#endif
    else {
        throw std::runtime_error("Unsupported device type for ATen");
    }
}

at::Tensor to_aten_tensor(const infinicore::Tensor &t);

#if defined(ENABLE_HYGON_API)
c10::hip::HIPStream get_hip_stream();
#elif defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
c10::cuda::CUDAStream get_cuda_stream();
#endif

#if defined(ENABLE_MOORE_API)
c10::musa::MUSAStream get_musa_stream();
#endif

} // namespace infinicore::adaptor

#endif // ENABLE_ATEN
