#include "../../handle.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/gemm.h"

#include <base/gemm.h>
#include <config.h>
#include <handle.h>
#include <operator.h>
#include <tensor.h>
#include <torch/ops/gemm/gemm.h>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace {

infini::ops::DataType dataTypeFromInfiniDtype(infiniDtype_t dtype) {
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
        return infini::ops::DataType::kFloat32;
    }
}

infini::ops::Device::Type deviceTypeFromInfiniDevice(infiniDevice_t device) {
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
        return infini::ops::Device::Type::kCpu;
    }
}

struct TensorMeta {
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
    infini::ops::DataType dtype;
};

TensorMeta makeTensorMeta(infiniopTensorDescriptor_t desc) {
    return TensorMeta{desc->shape(), desc->strides(), dataTypeFromInfiniDtype(desc->dtype())};
}

class InfiniOpsGemmDescriptor final : public InfiniopDescriptor {
public:
    InfiniOpsGemmDescriptor(infiniopHandle_t handle,
                            infiniopTensorDescriptor_t c_desc,
                            infiniopTensorDescriptor_t a_desc,
                            infiniopTensorDescriptor_t b_desc)
        : InfiniopDescriptor{handle->device, handle->device_id},
          device(deviceTypeFromInfiniDevice(handle->device), handle->device_id),
          c(makeTensorMeta(c_desc)),
          a(makeTensorMeta(a_desc)),
          b(makeTensorMeta(b_desc)) {}

    infini::ops::Tensor tensor(const TensorMeta &meta, void *data) const {
        return infini::ops::Tensor(data, meta.shape, meta.dtype, device, meta.strides);
    }

    infini::ops::Tensor tensor(const TensorMeta &meta, const void *data) const {
        return tensor(meta, const_cast<void *>(data));
    }

    infini::ops::Device device;
    TensorMeta c;
    TensorMeta a;
    TensorMeta b;
};

bool isExplicitTorchGemmDevice(infiniDevice_t device) {
    switch (device) {
    case INFINI_DEVICE_CPU:
    case INFINI_DEVICE_NVIDIA:
    case INFINI_DEVICE_CAMBRICON:
    case INFINI_DEVICE_ASCEND:
    case INFINI_DEVICE_METAX:
    case INFINI_DEVICE_MOORE:
    case INFINI_DEVICE_ILUVATAR:
    case INFINI_DEVICE_KUNLUN:
    case INFINI_DEVICE_HYGON:
    case INFINI_DEVICE_QY:
        return true;
    default:
        return false;
    }
}

} // namespace

__INFINI_C infiniStatus_t infiniopCreateGemmDescriptor(infiniopHandle_t handle,
                                                       infiniopGemmDescriptor_t *desc_ptr,
                                                       infiniopTensorDescriptor_t c,
                                                       const infiniopTensorDescriptor_t a,
                                                       const infiniopTensorDescriptor_t b) {
    if (!isExplicitTorchGemmDevice(handle->device)) {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    *desc_ptr = new InfiniOpsGemmDescriptor(handle, c, a, b);
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C infiniStatus_t infiniopGetGemmWorkspaceSize(infiniopGemmDescriptor_t desc, size_t *size) {
    (void)desc;
    *size = 0;
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C infiniStatus_t infiniopGemm(infiniopGemmDescriptor_t desc,
                                       void *workspace,
                                       size_t workspace_size,
                                       void *c,
                                       const void *a,
                                       const void *b,
                                       float alpha,
                                       float beta,
                                       void *stream) {
    auto *gemm_desc = reinterpret_cast<InfiniOpsGemmDescriptor *>(desc);
    if (!isExplicitTorchGemmDevice(gemm_desc->device_type)) {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

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
    delete reinterpret_cast<InfiniOpsGemmDescriptor *>(desc);
    return INFINI_STATUS_SUCCESS;
}
