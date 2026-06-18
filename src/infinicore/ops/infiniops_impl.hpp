#pragma once

#include "../utils.hpp"
#include "infinicore/tensor.hpp"

#include <stdexcept>

#include "infini/operator_call_instantiations.h"
#include "tensor.h"

namespace infinicore::op::infiniops {

inline infini::ops::DataType toInfiniOpsDtype(DataType dtype) {
    switch (dtype) {
    case DataType::I8:
        return infini::ops::DataType::kInt8;
    case DataType::I16:
        return infini::ops::DataType::kInt16;
    case DataType::I32:
        return infini::ops::DataType::kInt32;
    case DataType::I64:
        return infini::ops::DataType::kInt64;
    case DataType::U8:
    case DataType::BYTE:
        return infini::ops::DataType::kUInt8;
    case DataType::U16:
        return infini::ops::DataType::kUInt16;
    case DataType::U32:
        return infini::ops::DataType::kUInt32;
    case DataType::U64:
        return infini::ops::DataType::kUInt64;
    case DataType::F16:
        return infini::ops::DataType::kFloat16;
    case DataType::BF16:
        return infini::ops::DataType::kBFloat16;
    case DataType::F32:
        return infini::ops::DataType::kFloat32;
    case DataType::F64:
        return infini::ops::DataType::kFloat64;
    default:
        throw std::runtime_error("InfiniOps backend does not support this tensor dtype.");
    }
}

inline infini::ops::Device toInfiniOpsDevice(const Device &device) {
    INFINICORE_ASSERT(device.getType() == Device::Type::NVIDIA);
    return infini::ops::Device{infini::ops::Device::Type::kNvidia, static_cast<int>(device.getIndex())};
}

struct TensorMeta {
    Shape shape;
    Strides strides;
    infini::ops::DataType dtype;
    infini::ops::Device device;

    explicit TensorMeta(const Tensor &tensor)
        : shape(tensor->shape()),
          strides(tensor->strides()),
          dtype(toInfiniOpsDtype(tensor->dtype())),
          device(toInfiniOpsDevice(tensor->device())) {}

    infini::ops::Tensor tensor(const void *data) const {
        return infini::ops::Tensor(
            const_cast<void *>(data), shape, dtype, device, strides);
    }

    infini::ops::Tensor tensor(const Tensor &tensor) const {
        return this->tensor(tensor->data());
    }
};

} // namespace infinicore::op::infiniops
