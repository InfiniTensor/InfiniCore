#include "kv_caching_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

namespace op::kv_caching::ascend {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_cache,
    infiniopTensorDescriptor_t v_cache,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t v,
    infiniopTensorDescriptor_t past_kv_lengths) {
    auto info = KVCachingInfo::createKVCachingInfo(k_cache, v_cache, k, v, past_kv_lengths);
    CHECK_RESULT(info);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{},
        info.take(), 0, handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *k_cache,
    void *v_cache,
    const void *k,
    const void *v,
    const void *past_kv_lengths,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;

    return kv_caching_kernel_launch(
        k_cache, v_cache, k, v, past_kv_lengths,
        _info.dtype, _info.past_len_dtype,
        _info.batch_size, _info.num_kv_heads, _info.seq_len, _info.hidden_dim,
        _info.k_cache_strides_0, _info.k_cache_strides_1, _info.k_cache_strides_2, _info.k_cache_strides_3,
        _info.v_cache_strides_0, _info.v_cache_strides_1, _info.v_cache_strides_2, _info.v_cache_strides_3,
        _info.k_strides_0, _info.k_strides_1, _info.k_strides_2, _info.k_strides_3,
        _info.v_strides_0, _info.v_strides_1, _info.v_strides_2, _info.v_strides_3,
        stream);
}

} // namespace op::kv_caching::ascend
