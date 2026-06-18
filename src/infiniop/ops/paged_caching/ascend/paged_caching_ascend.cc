#include "paged_caching_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

namespace op::paged_caching::ascend {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slot_mapping_desc) {

    auto info = PagedCachingInfo::create(k_cache_desc, v_cache_desc, k_desc, v_desc, slot_mapping_desc);
    CHECK_RESULT(info);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{},
        info.take(),
        0,
        handle_ascend->device,
        handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *k_cache, void *v_cache,
    const void *k, const void *v,
    const void *slot_mapping,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;

    return paged_caching_kernel_launch(
        k_cache, v_cache, k, v, slot_mapping,
        _info.dtype,
        _info.num_tokens,
        _info.num_kv_heads,
        _info.head_size,
        _info.block_size,
        _info.k_src_stride,
        _info.v_src_stride,
        _info.k_cache_block_stride,
        _info.v_cache_block_stride,
        _info.k_cache_head_stride,
        _info.v_cache_head_stride,
        _info.k_cache_slot_stride,
        _info.v_cache_slot_stride,
        stream);
}

} // namespace op::paged_caching::ascend
