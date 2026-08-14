#include "rms_norm_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_rms_norm.h>

extern "C" infiniStatus_t rms_norm_cast_w_launch(
    void *dst, const void *src,
    infiniDtype_t src_dtype, infiniDtype_t dst_dtype,
    size_t count, void *stream);

namespace op::rms_norm::ascend {
namespace {

bool isContiguous(const std::vector<size_t> &shape,
                  const std::vector<ptrdiff_t> &strides) {
    ptrdiff_t expected_stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        if (strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<ptrdiff_t>(shape[i]);
    }
    return true;
}

} // namespace

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t y;
    aclnnTensorDescriptor_t x;
    aclnnTensorDescriptor_t w;
    aclnnTensorDescriptor_t rstd;
    size_t workspaceSize;
    aclOpExecutor *executor;
    bool needs_cast_w;
    bool uses_sliced_execution;
    size_t cast_w_offset;
    size_t w_padded_offset;
    size_t w_padded_size;

    Opaque(aclnnTensorDescriptor_t y_, aclnnTensorDescriptor_t x_,
           aclnnTensorDescriptor_t w_, aclnnTensorDescriptor_t rstd_,
           size_t ws, aclOpExecutor *exec,
           bool cast_w, bool sliced_execution,
           size_t cast_off, size_t pad_off, size_t pad_sz)
        : y(y_), x(x_), w(w_), rstd(rstd_), workspaceSize(ws), executor(exec),
          needs_cast_w(cast_w), uses_sliced_execution(sliced_execution),
          cast_w_offset(cast_off),
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

    // aclnnRmsNorm writes output as contiguous storage even when the output
    // descriptor has non-contiguous outer strides. Keep the full-tensor fast
    // path for contiguous output, and use row slices only when layout requires it.
    bool uses_sliced_execution = !isContiguous(info.shape, info.y_strides);
    aclnnTensorDescriptor_t y = nullptr;
    aclnnTensorDescriptor_t x = nullptr;
    if (uses_sliced_execution) {
        std::vector<int64_t> slice_shape = {static_cast<int64_t>(info.dim())};
        std::vector<int64_t> slice_strides = {1};
        y = new aclnnTensorDescriptor(toAclDataType(info.atype), slice_shape, slice_strides);
        x = new aclnnTensorDescriptor(toAclDataType(info.atype), slice_shape, slice_strides);
    } else {
        y = new aclnnTensorDescriptor(y_desc);
        x = new aclnnTensorDescriptor(x_desc);
    }

    // 仅在跨半精度组合时需要将 w cast 到 atype
    // (F16 atype + BF16 w, 或 BF16 atype + F16 w)
    bool needs_cast_w = (info.atype != info.wtype && info.wtype != INFINI_DTYPE_F32);
    aclnnTensorDescriptor_t w = nullptr;
    if (needs_cast_w) {
        // 规避 constructor #2 的 ndim 内存 corruption 问题
        // 先用 constructor #1 从 w_desc 正确构造，再替换 tensor 为正确的 dtype
        w = new aclnnTensorDescriptor(w_desc);
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

    std::vector<int64_t> rstd_shape = {1};
    std::vector<int64_t> rstd_strides = {1};
    if (!uses_sliced_execution) {
        rstd_shape.clear();
        rstd_shape.reserve(info.ndim() - 1);
        for (size_t i = 0; i + 1 < info.ndim(); ++i) {
            rstd_shape.push_back(static_cast<int64_t>(info.shape[i]));
        }
        rstd_strides.assign(rstd_shape.size(), 1);
        for (ptrdiff_t i = static_cast<ptrdiff_t>(rstd_shape.size()) - 2; i >= 0; --i) {
            rstd_strides[i] = rstd_strides[i + 1] * rstd_shape[i + 1];
        }
    }
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
        new Opaque{y, x, w, rstd, workspace_size, executor,
                   needs_cast_w, uses_sliced_execution,
                   cast_w_offset, w_padded_offset, w_padded_size},
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

    void *rstd_ptr = static_cast<uint8_t *>(workspace) + _opaque->workspaceSize;
    void *w_ptr = nullptr;
    if (_opaque->needs_cast_w) {
        void *cast_w_ptr = static_cast<uint8_t *>(workspace) + _opaque->cast_w_offset;
        void *w_padded_src = static_cast<uint8_t *>(workspace) + _opaque->w_padded_offset;
        size_t w_bytes = _info.dim() * infiniSizeOf(_info.wtype);
        CHECK_ACL(aclrtMemcpyAsync(
            w_padded_src, _opaque->w_padded_size, const_cast<void *>(w), w_bytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE, static_cast<aclrtStream>(stream)));
        CHECK_STATUS(rms_norm_cast_w_launch(
            cast_w_ptr, w_padded_src, _info.wtype, INFINI_DTYPE_F32,
            _info.dim(), stream));
        w_ptr = cast_w_ptr;
    } else {
        w_ptr = const_cast<void *>(w);
    }

    CHECK_ACL(AclSetTensorAddr(_opaque->executor, 1, _opaque->w->tensor, w_ptr));
    CHECK_ACL(AclSetTensorAddr(_opaque->executor, 3, _opaque->rstd->tensor, rstd_ptr));

    if (!_opaque->uses_sliced_execution) {
        CHECK_ACL(AclSetTensorAddr(_opaque->executor, 0, _opaque->x->tensor, const_cast<void *>(x)));
        CHECK_ACL(AclSetTensorAddr(_opaque->executor, 2, _opaque->y->tensor, y));
        CHECK_ACL(aclnnRmsNorm(
            workspace, _opaque->workspaceSize, _opaque->executor, stream));
        return INFINI_STATUS_SUCCESS;
    }

    const size_t element_size = infiniSizeOf(_info.atype);
    const size_t batch = _info.shape[0];
    const size_t nhead = _info.ndim() == 3 ? _info.shape[1] : 1;
    for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        for (size_t head_idx = 0; head_idx < nhead; ++head_idx) {
            ptrdiff_t x_offset = batch_idx * _info.x_strides[0];
            ptrdiff_t y_offset = batch_idx * _info.y_strides[0];
            if (_info.ndim() == 3) {
                x_offset += head_idx * _info.x_strides[1];
                y_offset += head_idx * _info.y_strides[1];
            }

            auto x_row = static_cast<const uint8_t *>(x) + x_offset * element_size;
            auto y_row = static_cast<uint8_t *>(y) + y_offset * element_size;
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 0, _opaque->x->tensor,
                const_cast<uint8_t *>(x_row)));
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 2, _opaque->y->tensor, y_row));
            CHECK_ACL(aclnnRmsNorm(
                workspace, _opaque->workspaceSize, _opaque->executor, stream));
        }
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::ascend
