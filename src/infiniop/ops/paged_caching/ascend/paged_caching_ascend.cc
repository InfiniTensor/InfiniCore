#include "paged_caching_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_scatter_pa_kv_cache.h>
#include <memory>

namespace op::paged_caching::ascend {

struct Descriptor::Opaque {
    bool aclnn = false;
    std::unique_ptr<aclnnTensorDescriptor> k, v, k_cache, v_cache, slots;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slot_mapping_desc) {
    auto info = PagedCachingInfo::create(k_cache_desc, v_cache_desc, k_desc, v_desc, slot_mapping_desc);
    CHECK_RESULT(info);
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    *desc_ptr = new Descriptor(new Opaque{}, info.take(), 0,
                              handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::createWithValidSlots(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slot_mapping_desc) {
    auto status = create(handle, desc_ptr, k_cache_desc, v_cache_desc,
                         k_desc, v_desc, slot_mapping_desc);
    if (status != INFINI_STATUS_SUCCESS) { return status; }
    auto *d = *desc_ptr;
    const auto &i = d->_info;
    // InfiniOP exposes BHSD strides. ACLNN Norm consumes physically contiguous
    // BSHD, so reinterpret only the exact compatible layout without a copy.
    const bool compatible = k_cache_desc->ndim() == 4 && v_cache_desc->ndim() == 4
        && k_desc->isContiguous() && v_desc->isContiguous()
        && slot_mapping_desc->isContiguous()
        && k_cache_desc->stride(3) == 1 && v_cache_desc->stride(3) == 1
        && i.k_cache_head_stride == static_cast<ptrdiff_t>(i.head_size)
        && i.v_cache_head_stride == static_cast<ptrdiff_t>(i.v_head_size)
        && i.k_cache_slot_stride == static_cast<ptrdiff_t>(i.num_kv_heads * i.head_size)
        && i.v_cache_slot_stride == static_cast<ptrdiff_t>(i.num_kv_heads * i.v_head_size)
        && i.k_cache_block_stride == static_cast<ptrdiff_t>(i.block_size * i.num_kv_heads * i.head_size)
        && i.v_cache_block_stride == static_cast<ptrdiff_t>(i.block_size * i.num_kv_heads * i.v_head_size)
        && i.num_tokens > 0;
    if (!compatible) { return INFINI_STATUS_SUCCESS; }
    auto *p = d->_opaque;
    const auto dtype = toAclDataType(i.dtype);
    const int64_t blocks = static_cast<int64_t>(k_cache_desc->dim(0));
    const int64_t bs = static_cast<int64_t>(i.block_size);
    const int64_t heads = static_cast<int64_t>(i.num_kv_heads);
    const int64_t kd = static_cast<int64_t>(i.head_size);
    const int64_t vd = static_cast<int64_t>(i.v_head_size);
    p->k = std::make_unique<aclnnTensorDescriptor>(k_desc);
    p->v = std::make_unique<aclnnTensorDescriptor>(v_desc);
    p->slots = std::make_unique<aclnnTensorDescriptor>(slot_mapping_desc);
    p->k_cache = std::make_unique<aclnnTensorDescriptor>(dtype,
        std::vector<int64_t>{blocks, bs, heads, kd},
        std::vector<int64_t>{bs * heads * kd, heads * kd, kd, 1});
    p->v_cache = std::make_unique<aclnnTensorDescriptor>(dtype,
        std::vector<int64_t>{blocks, bs, heads, vd},
        std::vector<int64_t>{bs * heads * vd, heads * vd, vd, 1});
    aclOpExecutor *executor = nullptr;
    uint64_t bytes = 0;
    const auto ret = aclnnScatterPaKvCacheGetWorkspaceSize(
        p->k->tensor, p->k_cache->tensor, p->slots->tensor,
        p->v->tensor, p->v_cache->tensor, nullptr, nullptr, nullptr,
        const_cast<char *>("Norm"), const_cast<char *>("None"), nullptr, nullptr,
        &bytes, &executor);
    // This Nnopbase executor does not support Repeatable; its lifecycle is CANN-managed.
    if (ret != ACL_SUCCESS) {
        delete d;
        *desc_ptr = nullptr;
        CHECK_ACL(ret);
    }
    d->_workspace_size = bytes;
    p->aclnn = true;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *k_cache, void *v_cache, const void *k, const void *v,
    const void *slot_mapping, void *stream) const {
    if (_opaque->aclnn) {
        const auto *p = _opaque;
        auto view = [](const std::unique_ptr<aclnnTensorDescriptor> &meta, void *data) {
            return std::make_unique<aclnnTensorDescriptor>(meta->dataType, meta->shape, meta->strides, data);
        };
        auto ka = view(p->k, const_cast<void *>(k));
        auto va = view(p->v, const_cast<void *>(v));
        auto kc = view(p->k_cache, k_cache);
        auto vc = view(p->v_cache, v_cache);
        auto slots = view(p->slots, const_cast<void *>(slot_mapping));
        aclOpExecutor *executor = nullptr;
        uint64_t bytes = 0;
        CHECK_ACL(aclnnScatterPaKvCacheGetWorkspaceSize(
            ka->tensor, kc->tensor, slots->tensor, va->tensor, vc->tensor,
            nullptr, nullptr, nullptr, const_cast<char *>("Norm"), const_cast<char *>("None"),
            nullptr, nullptr, &bytes, &executor));
        if (bytes > workspace_size) {
            // This Nnopbase executor does not support Repeatable; its lifecycle is CANN-managed.
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        CHECK_ACL(aclnnScatterPaKvCache(workspace, bytes, executor, stream));
        return INFINI_STATUS_SUCCESS;
    }
    return paged_caching_kernel_launch(
        k_cache, v_cache, k, v, slot_mapping, _info.dtype,
        _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size,
        _info.block_size, _info.k_src_stride, _info.v_src_stride,
        _info.k_src_head_stride, _info.v_src_head_stride,
        _info.k_cache_block_stride, _info.v_cache_block_stride,
        _info.k_cache_head_stride, _info.v_cache_head_stride,
        _info.k_cache_slot_stride, _info.v_cache_slot_stride, stream);
}

} // namespace op::paged_caching::ascend
