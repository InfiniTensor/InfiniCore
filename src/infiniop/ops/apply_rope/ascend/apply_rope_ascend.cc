#include "apply_rope_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_apply_rotary_pos_emb.h>
#include <array>
#include <cstdint>
#include <mutex>
#include <utility>
namespace op::apply_rope::ascend {
struct Descriptor::Opaque {
    std::array<std::unique_ptr<aclnnTensorDescriptor>, 4> tensors;
    aclOpExecutor *executor = nullptr;
    int device_id = -1;
    void *stream = nullptr;
    bool stream_set = false;
    size_t q_bytes = 0, k_bytes = 0;
    std::mutex mutex;
    ~Opaque() {
        if (executor) {
            int previous = -1;
            if (stream_set && aclrtGetDevice(&previous) == ACL_SUCCESS
                && aclrtSetDevice(device_id) == ACL_SUCCESS) {
                aclrtSynchronizeStream(stream);
                aclrtSetDevice(previous);
            }
            aclDestroyAclOpExecutor(executor);
        }
    }
};
Descriptor::Descriptor(std::unique_ptr<Opaque> p, size_t bytes,
                       infiniDevice_t device, int id)
    : InfiniopDescriptor{device, id}, opaque_(std::move(p)), workspace_size_(bytes) {}
Descriptor::~Descriptor() = default;
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t cos, infiniopTensorDescriptor_t sin) {
    const auto dtype = q->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    for (auto t : {q, k, cos, sin}) {
        if (t->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (t->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        // The installed CANN repeatable path crashes for strided references.
        if (!t->isContiguous()) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
    }
    const auto T = q->dim(0), D = q->dim(2);
    if (T == 0 || q->dim(1) == 0 || k->dim(1) == 0
        || (D != 64 && D != 128) || k->dim(0) != T || k->dim(2) != D
        || cos->dim(0) != T || cos->dim(1) != 1 || cos->dim(2) != D
        || sin->shape() != cos->shape()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    auto p = std::make_unique<Opaque>();
    const std::array<infiniopTensorDescriptor_t, 4> descs{q, k, cos, sin};
    for (size_t i = 0; i < descs.size(); ++i) {
        p->tensors[i] = std::make_unique<aclnnTensorDescriptor>(descs[i]);
    }
    uint64_t bytes = 0;
    CHECK_ACL(aclnnApplyRotaryPosEmbGetWorkspaceSize(
        p->tensors[0]->tensor, p->tensors[1]->tensor,
        p->tensors[2]->tensor, p->tensors[3]->tensor, 4,
        &bytes, &p->executor));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(p->executor));
    p->device_id = handle->device_id;
    p->q_bytes = T * q->dim(1) * D * infiniSizeOf(dtype);
    p->k_bytes = T * k->dim(1) * D * infiniSizeOf(dtype);
    *desc_ptr = new Descriptor(std::move(p), bytes, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size, void *q, void *k,
    const void *cos, const void *sin, void *stream) const {
    if (workspace_size < workspace_size_) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    const auto qa = reinterpret_cast<uintptr_t>(q), ka = reinterpret_cast<uintptr_t>(k);
    if (qa < ka + opaque_->k_bytes && ka < qa + opaque_->q_bytes) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    std::lock_guard<std::mutex> lock(opaque_->mutex);
    if (!opaque_->stream_set) {
        opaque_->stream = stream;
        opaque_->stream_set = true;
    }
    if (opaque_->stream != stream) {
        // Keep the retained executor on its first stream. Other streams use
        // independent one-shot executors, including independent tensor refs.
        const std::array<const void *, 4> addresses{q, k, cos, sin};
        std::array<std::unique_ptr<aclnnTensorDescriptor>, 4> call;
        for (size_t i = 0; i < call.size(); ++i) {
            const auto &meta = *opaque_->tensors[i];
            call[i] = std::make_unique<aclnnTensorDescriptor>(
                meta.dataType, meta.shape, meta.strides, meta.format,
                meta.storageShape, const_cast<void *>(addresses[i]));
        }
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnApplyRotaryPosEmbGetWorkspaceSize(
            call[0]->tensor, call[1]->tensor, call[2]->tensor, call[3]->tensor,
            4, &bytes, &executor));
        if (workspace_size < bytes) {
            aclDestroyAclOpExecutor(executor);
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        CHECK_ACL(aclnnApplyRotaryPosEmb(workspace, bytes, executor, stream));
        return INFINI_STATUS_SUCCESS;
    }
    const std::array<const void *, 4> inputs{q, k, cos, sin};
    for (size_t i = 0; i < inputs.size(); ++i) {
        CHECK_ACL(aclSetInputTensorAddr(opaque_->executor, i,
                                        opaque_->tensors[i]->tensor, const_cast<void *>(inputs[i])));
    }
    CHECK_ACL(aclSetOutputTensorAddr(opaque_->executor, 0, opaque_->tensors[0]->tensor, q));
    CHECK_ACL(aclSetOutputTensorAddr(opaque_->executor, 1, opaque_->tensors[1]->tensor, k));
    CHECK_ACL(aclnnApplyRotaryPosEmb(workspace, workspace_size_, opaque_->executor, stream));
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::apply_rope::ascend
