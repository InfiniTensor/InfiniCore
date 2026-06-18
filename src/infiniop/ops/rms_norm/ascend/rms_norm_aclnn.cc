#include "rms_norm_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_rms_norm.h>
#include <cstdio>

extern "C" infiniStatus_t rms_norm_cast_w_launch(
    void *dst, const void *src,
    infiniDtype_t src_dtype, infiniDtype_t dst_dtype,
    size_t count, void *stream);

namespace op::rms_norm::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t y;
    aclnnTensorDescriptor_t x;
    aclnnTensorDescriptor_t w;
    aclnnTensorDescriptor_t rstd;
    size_t workspaceSize;
    aclOpExecutor *executor;
    bool needs_cast_w;
    size_t cast_w_offset;
    size_t w_padded_offset;
    size_t w_padded_size;

    Opaque(aclnnTensorDescriptor_t y_, aclnnTensorDescriptor_t x_,
           aclnnTensorDescriptor_t w_, aclnnTensorDescriptor_t rstd_,
           size_t ws, aclOpExecutor *exec,
           bool cast_w, size_t cast_off, size_t pad_off, size_t pad_sz)
        : y(y_), x(x_), w(w_), rstd(rstd_), workspaceSize(ws), executor(exec),
          needs_cast_w(cast_w), cast_w_offset(cast_off),
          w_padded_offset(pad_off), w_padded_size(pad_sz) {}

    ~Opaque() {
        delete y;
        delete x;
        delete w;
        delete rstd;
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

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    std::vector<int64_t> slice_shape = {static_cast<int64_t>(info.dim())};
    auto slice_stride = std::vector<int64_t>(1, 1);

    aclnnTensorDescriptor_t y = new aclnnTensorDescriptor(toAclDataType(info.atype), slice_shape, slice_stride);
    aclnnTensorDescriptor_t x = new aclnnTensorDescriptor(toAclDataType(info.atype), slice_shape, slice_stride);

    // 仅在跨半精度组合时需要将 w cast 到 atype
    // (F16 atype + BF16 w, 或 BF16 atype + F16 w)
    bool needs_cast_w = (info.atype != info.wtype && info.wtype != INFINI_DTYPE_F32);
    aclnnTensorDescriptor_t w = nullptr;
    std::vector<int64_t> w_shape_i64_dbg;
    std::vector<int64_t> w_strides_i64_dbg;
    if (needs_cast_w) {
        // 规避 constructor #2 的 ndim 内存 corruption 问题
        // 先用 constructor #1 从 w_desc 正确构造，再替换 tensor 为正确的 dtype
        w = new aclnnTensorDescriptor(w_desc);
        w_shape_i64_dbg = w->shape;
        w_strides_i64_dbg = w->strides;
        if (w->tensor) {
            aclDestroyTensor(w->tensor);
        }
        w->dataType = toAclDataType(INFINI_DTYPE_F32);
        w->tensor = aclCreateTensor(w->shape.data(), w->ndim, w->dataType,
                                    w->strides.data(), w->offset, w->format,
                                    w->storageShape.data(), w->storageNdim, nullptr);
    } else {
        w = new aclnnTensorDescriptor(w_desc);
    }

    auto rstd_shape = std::vector<int64_t>(1, 1);
    auto rstd_strides = std::vector<int64_t>(1, 1);
    aclnnTensorDescriptor_t rstd = new aclnnTensorDescriptor(toAclDataType(INFINI_DTYPE_F32), rstd_shape, rstd_strides);

    size_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    CHECK_ACL(aclnnRmsNormGetWorkspaceSize(
        x->tensor,
        w->tensor,
        static_cast<double>(epsilon),
        y->tensor,
        rstd->tensor,
        &workspace_size,
        &executor));

    aclSetAclOpExecutorRepeatable(executor);

    size_t rstd_size = rstd->numel() * aclDataTypeSize(rstd->dataType);
    size_t cast_w_dst_size = needs_cast_w ? info.dim() * sizeof(float) : 0;
    size_t w_padded_size = 0;
    if (needs_cast_w) {
        size_t w_raw_bytes = info.dim() * infiniSizeOf(info.wtype);
        w_padded_size = ((w_raw_bytes + 31) / 32) * 32;
    }
    size_t all_workspace_size = workspace_size + rstd_size + cast_w_dst_size + w_padded_size;
    size_t cast_w_offset = workspace_size + rstd_size;
    size_t w_padded_offset = cast_w_offset + cast_w_dst_size;

    *desc_ptr = new Descriptor(
        new Opaque{y, x, w, rstd, workspace_size, executor, needs_cast_w, cast_w_offset, w_padded_offset, w_padded_size},
        std::move(info),
        all_workspace_size,
        handle_ascend->device,
        handle_ascend->device_id);

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

    void *rstdPtr = (void *)((uint8_t *)workspace + _opaque->workspaceSize);
    void *w_ptr = nullptr;

    if (_opaque->needs_cast_w) {
        void *cast_w_ptr = (void *)((uint8_t *)workspace + _opaque->cast_w_offset);
        void *w_padded_src = (void *)((uint8_t *)workspace + _opaque->w_padded_offset);
        size_t w_bytes = _info.dim() * infiniSizeOf(_info.wtype);
        aclrtMemcpyAsync(w_padded_src, _opaque->w_padded_size, (void *)w, w_bytes,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, (aclrtStream)stream);
        rms_norm_cast_w_launch(cast_w_ptr, w_padded_src, _info.wtype, INFINI_DTYPE_F32, _info.dim(), stream);
        w_ptr = cast_w_ptr;
    } else {
        w_ptr = (void *)w;
    }

    auto unit = infiniSizeOf(_info.atype);

    AclSetTensorAddr(_opaque->executor, 1, tw, w_ptr);
    AclSetTensorAddr(_opaque->executor, 3, trstd, rstdPtr);

    auto ndim = _info.ndim();
    size_t outer = ndim == 2 ? 1 : _info.shape[0];
    size_t inner = ndim == 2 ? _info.shape[0] : _info.shape[1];

    for (size_t b = 0; b < outer; ++b) {
        for (size_t s = 0; s < inner; ++s) {
            ptrdiff_t x_offset, y_offset;
            if (ndim == 2) {
                x_offset = s * _info.x_strides[0];
                y_offset = s * _info.y_strides[0];
            } else {
                x_offset = b * _info.x_strides[0] + s * _info.x_strides[1];
                y_offset = b * _info.y_strides[0] + s * _info.y_strides[1];
            }
            AclSetTensorAddr(_opaque->executor, 0, tx, ((char *)x) + x_offset * unit);
            AclSetTensorAddr(_opaque->executor, 2, ty, ((char *)y) + y_offset * unit);
            CHECK_ACL(aclnnRmsNorm(workspace, _opaque->workspaceSize, _opaque->executor, stream));
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::ascend
