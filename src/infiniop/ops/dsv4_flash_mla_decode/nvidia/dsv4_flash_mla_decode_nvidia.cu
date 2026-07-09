#include "../../../handle.h"
#include "dsv4_flash_mla_decode_nvidia.cuh"

#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <tvm/ffi/container/tensor.h>

namespace {
using flash_mla_q_nope_pe_fn_t = void (*)(
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    long,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView,
    double,
    bool,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView);

constexpr const char *kFlashMlaQNopePeSymbol = "_Z36flash_mla_with_kvcache_q_nope_pe_outN3tvm3ffi10TensorViewES1_S1_S1_S1_lS1_S1_dbS1_S1_";

const char *deepseekOpsPath() {
    const char *env = std::getenv("DEEPSEEK_V4_OPS_SO");
    if (env && env[0] != '\0') {
        return env;
    }
    return "libdeepseek_v4_ops.so";
}

void *tryDlopen(const char *path, int flags) {
    dlerror();
    return dlopen(path, flags);
}

flash_mla_q_nope_pe_fn_t resolveFlashMlaQNopePe() {
    static flash_mla_q_nope_pe_fn_t fn = []() -> flash_mla_q_nope_pe_fn_t {
        tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
        tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL);
        void *handle = tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            return nullptr;
        }
        return reinterpret_cast<flash_mla_q_nope_pe_fn_t>(dlsym(handle, kFlashMlaQNopePeSymbol));
    }();
    return fn;
}

DLDataType toDlDtype(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        return DLDataType{kDLFloat, 16, 1};
    case INFINI_DTYPE_BF16:
        return DLDataType{kDLBfloat, 16, 1};
    case INFINI_DTYPE_F32:
        return DLDataType{kDLFloat, 32, 1};
    case INFINI_DTYPE_I32:
        return DLDataType{kDLInt, 32, 1};
    default:
        return DLDataType{kDLOpaqueHandle, 0, 1};
    }
}

template <size_t N>
tvm::ffi::TensorView makeTensorView(const void *data,
                                    infiniDtype_t dtype,
                                    const std::array<int64_t, N> &shape,
                                    const std::array<int64_t, N> &strides,
                                    int device_id,
                                    DLTensor &tensor) {
    tensor.data = const_cast<void *>(data);
    tensor.device = DLDevice{kDLROCM, device_id};
    tensor.ndim = static_cast<int32_t>(N);
    tensor.dtype = toDlDtype(dtype);
    tensor.shape = const_cast<int64_t *>(shape.data());
    tensor.strides = const_cast<int64_t *>(strides.data());
    tensor.byte_offset = 0;
    return tvm::ffi::TensorView(&tensor);
}

} // namespace
#endif

namespace op::dsv4_flash_mla_decode::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t lse_desc, infiniopTensorDescriptor_t q_nope_desc, infiniopTensorDescriptor_t q_pe_desc, infiniopTensorDescriptor_t k_cache_desc, infiniopTensorDescriptor_t block_table_desc, infiniopTensorDescriptor_t cache_seqlens_desc, infiniopTensorDescriptor_t tile_scheduler_metadata_desc, infiniopTensorDescriptor_t num_splits_desc, int head_dim_v, float softmax_scale, bool causal) {
    Info info;
    CHECK_STATUS(createInfo(&info, out_desc, lse_desc, q_nope_desc, q_pe_desc, k_cache_desc, block_table_desc, cache_seqlens_desc, tile_scheduler_metadata_desc, num_splits_desc, head_dim_v, softmax_scale, causal));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *out, void *lse, const void *q_nope, const void *q_pe, const void *k_cache, const void *block_table, const void *cache_seqlens, const void *tile_scheduler_metadata, const void *num_splits, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveFlashMlaQNopePe();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);

    const auto &i = _info;
    int device_id = this->device_id;

    std::array<int64_t, 4> q_nope_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.seqlen_q), static_cast<int64_t>(i.heads_q), static_cast<int64_t>(i.head_dim_nope)};
    std::array<int64_t, 4> q_nope_strides{static_cast<int64_t>(i.seqlen_q * i.heads_q * i.head_dim_nope), static_cast<int64_t>(i.heads_q * i.head_dim_nope), static_cast<int64_t>(i.head_dim_nope), 1};
    std::array<int64_t, 4> q_pe_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.seqlen_q), static_cast<int64_t>(i.heads_q), static_cast<int64_t>(i.head_dim_pe)};
    std::array<int64_t, 4> q_pe_strides{static_cast<int64_t>(i.seqlen_q * i.heads_q * i.head_dim_pe), static_cast<int64_t>(i.heads_q * i.head_dim_pe), static_cast<int64_t>(i.head_dim_pe), 1};
    std::array<int64_t, 4> k_shape{static_cast<int64_t>(i.num_blocks), static_cast<int64_t>(i.page_block_size), static_cast<int64_t>(i.heads_kv), static_cast<int64_t>(i.head_dim_nope + i.head_dim_pe)};
    std::array<int64_t, 4> k_strides{static_cast<int64_t>(i.page_block_size * i.heads_kv * (i.head_dim_nope + i.head_dim_pe)), static_cast<int64_t>(i.heads_kv * (i.head_dim_nope + i.head_dim_pe)), static_cast<int64_t>(i.head_dim_nope + i.head_dim_pe), 1};
    std::array<int64_t, 4> out_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.seqlen_q), static_cast<int64_t>(i.heads_q), static_cast<int64_t>(i.head_dim_v)};
    std::array<int64_t, 4> out_strides{static_cast<int64_t>(i.seqlen_q * i.heads_q * i.head_dim_v), static_cast<int64_t>(i.heads_q * i.head_dim_v), static_cast<int64_t>(i.head_dim_v), 1};
    std::array<int64_t, 3> lse_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.heads_q), static_cast<int64_t>(i.seqlen_q)};
    std::array<int64_t, 3> lse_strides{static_cast<int64_t>(i.heads_q * i.seqlen_q), static_cast<int64_t>(i.seqlen_q), 1};
    std::array<int64_t, 2> block_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.max_blocks_per_seq)};
    std::array<int64_t, 2> block_strides{static_cast<int64_t>(i.max_blocks_per_seq), 1};
    std::array<int64_t, 1> seqlens_shape{static_cast<int64_t>(i.batch)};
    std::array<int64_t, 1> seqlens_strides{1};
    std::array<int64_t, 2> meta_shape{static_cast<int64_t>(i.tile_meta_rows), 8};
    std::array<int64_t, 2> meta_strides{8, 1};
    std::array<int64_t, 1> splits_shape{static_cast<int64_t>(i.batch + 1)};
    std::array<int64_t, 1> splits_strides{1};

    DLTensor q_nope_t, q_pe_t, k_t, block_t, seqlens_t, meta_t, splits_t, out_t, lse_t;
    auto q_nope_v = makeTensorView(q_nope, i.dtype, q_nope_shape, q_nope_strides, device_id, q_nope_t);
    auto q_pe_v = makeTensorView(q_pe, i.dtype, q_pe_shape, q_pe_strides, device_id, q_pe_t);
    auto k_v = makeTensorView(k_cache, i.dtype, k_shape, k_strides, device_id, k_t);
    auto block_v = makeTensorView(block_table, INFINI_DTYPE_I32, block_shape, block_strides, device_id, block_t);
    auto seqlens_v = makeTensorView(cache_seqlens, INFINI_DTYPE_I32, seqlens_shape, seqlens_strides, device_id, seqlens_t);
    auto meta_v = makeTensorView(tile_scheduler_metadata, INFINI_DTYPE_I32, meta_shape, meta_strides, device_id, meta_t);
    auto splits_v = makeTensorView(num_splits, INFINI_DTYPE_I32, splits_shape, splits_strides, device_id, splits_t);
    auto out_v = makeTensorView(out, i.dtype, out_shape, out_strides, device_id, out_t);
    auto lse_v = makeTensorView(lse, INFINI_DTYPE_F32, lse_shape, lse_strides, device_id, lse_t);

    fn(q_nope_v, q_pe_v, k_v, block_v, seqlens_v, static_cast<long>(i.head_dim_v), meta_v, splits_v, static_cast<double>(i.softmax_scale), i.causal, out_v, lse_v);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}

} // namespace op::dsv4_flash_mla_decode::nvidia
