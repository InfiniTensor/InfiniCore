#include "rms_norm_aclnn.h"
#include "../../../../utils/custom_types.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_rms_norm.h>
#include <cstdint>
#include <vector>

namespace op::rms_norm::ascend {

namespace {

size_t alignOffset(size_t offset, size_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

bool needsWeightCast(infiniDtype_t atype, infiniDtype_t wtype) {
    return (atype == INFINI_DTYPE_F16 || atype == INFINI_DTYPE_BF16)
        && (wtype == INFINI_DTYPE_F16 || wtype == INFINI_DTYPE_BF16)
        && atype != wtype;
}

void castWeightToFloat(float *dst, const void *src, size_t count, infiniDtype_t wtype) {
    if (wtype == INFINI_DTYPE_F16) {
        auto src_t = reinterpret_cast<const fp16_t *>(src);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = utils::cast<float>(src_t[i]);
        }
    } else if (wtype == INFINI_DTYPE_BF16) {
        auto src_t = reinterpret_cast<const bf16_t *>(src);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = utils::cast<float>(src_t[i]);
        }
    }
}

} // namespace

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t y;
    aclnnTensorDescriptor_t x;
    aclnnTensorDescriptor_t w;
    aclnnTensorDescriptor_t rstd;
    size_t workspaceSize;
    size_t rstdOffset;
    void *weightAddr;
    aclOpExecutor *executor;
    bool cast_weight;

    ~Opaque() {
        delete y;
        delete x;
        delete w;
        delete rstd;

        aclrtFree(weightAddr);
        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {

    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    size_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnTensorDescriptor_t y = nullptr;
    aclnnTensorDescriptor_t x = nullptr;
    aclnnTensorDescriptor_t w = nullptr;
    aclnnTensorDescriptor_t rstd = nullptr;
    void *weight_addr = nullptr;

    std::vector<int64_t> slice_shape = {static_cast<int64_t>(info.dim())};
    auto slice_stride = std::vector<int64_t>(1, 1);
    y = new aclnnTensorDescriptor(toAclDataType(info.atype), slice_shape, slice_stride);
    x = new aclnnTensorDescriptor(toAclDataType(info.atype), slice_shape, slice_stride);
    auto cast_weight = needsWeightCast(info.atype, info.wtype);
    w = cast_weight
          ? new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_F32), slice_shape, slice_stride)
          : new aclnnTensorDescriptor(w_desc);

    // Get AclTensor
    aclTensor *ty = y->tensor;
    aclTensor *tx = x->tensor;
    aclTensor *tw = w->tensor;
    // Set rstdDesc
    // See: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnRmsNorm.md
    // rstdTensor cannot set nullptr in aclnn
    auto rstd_shape = std::vector<int64_t>(1, 1);
    auto rstd_strides = std::vector<int64_t>(1, 1);
    rstd = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_F32), rstd_shape, rstd_strides);
    aclTensor *trstd = rstd->tensor;

    // Get WorkspaceSize and set executor
    CHECK_ACL(aclnnRmsNormGetWorkspaceSize(tx, tw, static_cast<double>(epsilon), ty, trstd, &workspace_size, &executor));
    aclSetAclOpExecutorRepeatable(executor);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    auto rstd_size = rstd->numel() * aclDataTypeSize(rstd->dataType);
    auto rstd_offset = alignOffset(workspace_size, 32);
    auto weight_workspace = cast_weight ? info.dim() * infiniSizeOf(INFINI_DTYPE_F32) : 0;
    if (cast_weight) {
        CHECK_ACL(aclrtMalloc(&weight_addr, weight_workspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    size_t all_workspace_size = rstd_offset + rstd_size;
    *desc_ptr = new Descriptor(
        new Opaque{y, x, w, rstd, workspace_size, rstd_offset, weight_addr, executor, cast_weight},
        std::move(info),
        all_workspace_size,
        handle_ascend->device, handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w,
    void *stream) const {

    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto tw = _opaque->w->tensor;
    auto tx = _opaque->x->tensor;
    auto ty = _opaque->y->tensor;
    auto trstd = _opaque->rstd->tensor;

    void *rstdPtr = static_cast<void *>(static_cast<uint8_t *>(workspace) + _opaque->rstdOffset);
    auto unit = infiniSizeOf(_info.atype);
    void *weightPtr = const_cast<void *>(w);

    if (_opaque->cast_weight) {
        auto weightBytesIn = _info.dim() * infiniSizeOf(_info.wtype);
        auto weightBytesOut = _info.dim() * infiniSizeOf(INFINI_DTYPE_F32);
        std::vector<uint8_t> hostWeightIn(weightBytesIn);
        std::vector<float> hostWeightOut(_info.dim());
        CHECK_ACL(aclrtMemcpy(hostWeightIn.data(), weightBytesIn, w, weightBytesIn, ACL_MEMCPY_DEVICE_TO_HOST));
        castWeightToFloat(hostWeightOut.data(), hostWeightIn.data(), _info.dim(), _info.wtype);
        weightPtr = _opaque->weightAddr;
        CHECK_ACL(aclrtMemcpy(weightPtr, weightBytesOut, hostWeightOut.data(), weightBytesOut, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    size_t outer = 1;
    for (size_t i = 0; i + 1 < _info.ndim(); ++i) {
        outer *= _info.shape[i];
    }

    AclSetTensorAddr(_opaque->executor, 1, tw, weightPtr);
    AclSetTensorAddr(_opaque->executor, 3, trstd, rstdPtr);
    for (size_t i = 0; i < outer; ++i) {
        size_t batch = _info.ndim() == 3 ? i / _info.shape[1] : i;
        size_t head = _info.ndim() == 3 ? i % _info.shape[1] : 0;
        auto x_offset = batch * _info.x_strides[0] + (_info.ndim() == 3 ? head * _info.x_strides[1] : 0);
        auto y_offset = batch * _info.y_strides[0] + (_info.ndim() == 3 ? head * _info.y_strides[1] : 0);
        AclSetTensorAddr(_opaque->executor, 0, tx, const_cast<char *>(static_cast<const char *>(x) + x_offset * unit));
        AclSetTensorAddr(_opaque->executor, 2, ty, static_cast<char *>(y) + y_offset * unit);
        CHECK_ACL(aclnnRmsNorm(workspace, _opaque->workspaceSize, _opaque->executor, stream));
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::ascend
