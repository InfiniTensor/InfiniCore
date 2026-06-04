#include "../../handle.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/gemm.h"

#include <base/gemm.h>
#include <config.h>
#include <data_type.h>
#include <device.h>
#include <native/cpu/device_.h>
#include <native/cuda/nvidia/device_.h>
#include <handle.h>
#include <operator.h>
#include <tensor.h>
#include <torch/ops/gemm/gemm.h>

#include <optional>
#include <vector>

namespace {

std::optional<infini::ops::DataType> toInfiniOpsDtype(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_I8:
        return infini::ops::DataType::kInt8;
    case INFINI_DTYPE_I16:
        return infini::ops::DataType::kInt16;
    case INFINI_DTYPE_I32:
        return infini::ops::DataType::kInt32;
    case INFINI_DTYPE_I64:
        return infini::ops::DataType::kInt64;
    case INFINI_DTYPE_U8:
        return infini::ops::DataType::kUInt8;
    case INFINI_DTYPE_U16:
        return infini::ops::DataType::kUInt16;
    case INFINI_DTYPE_U32:
        return infini::ops::DataType::kUInt32;
    case INFINI_DTYPE_U64:
        return infini::ops::DataType::kUInt64;
    case INFINI_DTYPE_F16:
        return infini::ops::DataType::kFloat16;
    case INFINI_DTYPE_BF16:
        return infini::ops::DataType::kBFloat16;
    case INFINI_DTYPE_F32:
        return infini::ops::DataType::kFloat32;
    case INFINI_DTYPE_F64:
        return infini::ops::DataType::kFloat64;
    default:
        return std::nullopt;
    }
}

std::optional<infini::ops::Device::Type> toInfiniOpsDevice(infiniDevice_t device) {
    switch (device) {
    case INFINI_DEVICE_CPU:
        return infini::ops::Device::Type::kCpu;
    case INFINI_DEVICE_NVIDIA:
        return infini::ops::Device::Type::kNvidia;
    case INFINI_DEVICE_CAMBRICON:
        return infini::ops::Device::Type::kCambricon;
    case INFINI_DEVICE_ASCEND:
        return infini::ops::Device::Type::kAscend;
    case INFINI_DEVICE_METAX:
        return infini::ops::Device::Type::kMetax;
    case INFINI_DEVICE_MOORE:
        return infini::ops::Device::Type::kMoore;
    case INFINI_DEVICE_ILUVATAR:
        return infini::ops::Device::Type::kIluvatar;
    case INFINI_DEVICE_KUNLUN:
        return infini::ops::Device::Type::kKunlun;
    case INFINI_DEVICE_HYGON:
        return infini::ops::Device::Type::kHygon;
    case INFINI_DEVICE_QY:
        return infini::ops::Device::Type::kQy;
    default:
        return std::nullopt;
    }
}

struct TensorMeta {
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
    infini::ops::DataType dtype;
};

TensorMeta makeMeta(infiniopTensorDescriptor_t desc, infini::ops::DataType dtype) {
    return TensorMeta{desc->shape(), desc->strides(), dtype};
}

struct InfiniOpsGemmDescriptor : InfiniopDescriptor {
    TensorMeta c;
    TensorMeta a;
    TensorMeta b;

    InfiniOpsGemmDescriptor(const InfiniopHandle *handle,
                            infiniopTensorDescriptor_t c_desc,
                            infiniopTensorDescriptor_t a_desc,
                            infiniopTensorDescriptor_t b_desc,
                            infini::ops::DataType c_dtype,
                            infini::ops::DataType a_dtype,
                            infini::ops::DataType b_dtype)
        : c(makeMeta(c_desc, c_dtype)), a(makeMeta(a_desc, a_dtype)), b(makeMeta(b_desc, b_dtype)) {
        device_type = handle->device;
        device_id = handle->device_id;
    }

    infini::ops::Tensor tensor(const TensorMeta &meta, const void *data) const {
        auto dev = toInfiniOpsDevice(device_type);
        if (!dev.has_value()) {
            return infini::ops::Tensor(const_cast<void *>(data), meta.shape, meta.dtype);
        }
        return infini::ops::Tensor(
            const_cast<void *>(data),
            meta.shape,
            meta.dtype,
            infini::ops::Device(*dev, device_id),
            meta.strides);
    }
};

} // namespace

__INFINI_C infiniStatus_t infiniopCreateGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    if (!toInfiniOpsDevice(handle->device).has_value()) {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    auto c_dtype = toInfiniOpsDtype(c_desc->dtype());
    auto a_dtype = toInfiniOpsDtype(a_desc->dtype());
    auto b_dtype = toInfiniOpsDtype(b_desc->dtype());
    if (!c_dtype.has_value() || !a_dtype.has_value() || !b_dtype.has_value()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    *desc_ptr = new InfiniOpsGemmDescriptor(handle, c_desc, a_desc, b_desc, *c_dtype, *a_dtype, *b_dtype);
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C infiniStatus_t infiniopGetGemmWorkspaceSize(
    infiniopGemmDescriptor_t,
    size_t *size) {
    *size = 0;
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C infiniStatus_t infiniopGemm(
    infiniopGemmDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    float alpha,
    float beta,
    void *stream) {
    auto gemm_desc = reinterpret_cast<const InfiniOpsGemmDescriptor *>(desc);

    infini::ops::Handle handle;
    handle.set_stream(stream);
    handle.set_workspace(workspace);
    handle.set_workspace_size_in_bytes(workspace_size);

    infini::ops::Config config;
    config.set_implementation_index(2);

    infini::ops::Operator<infini::ops::Gemm>::Call(
        handle,
        config,
        gemm_desc->tensor(gemm_desc->a, a),
        gemm_desc->tensor(gemm_desc->b, b),
        std::optional<float>(alpha),
        std::optional<float>(beta),
        std::optional<int>{},
        std::optional<int>{},
        gemm_desc->tensor(gemm_desc->c, c));
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C infiniStatus_t infiniopDestroyGemmDescriptor(infiniopGemmDescriptor_t desc) {
    delete reinterpret_cast<const InfiniOpsGemmDescriptor *>(desc);
    return INFINI_STATUS_SUCCESS;
}
