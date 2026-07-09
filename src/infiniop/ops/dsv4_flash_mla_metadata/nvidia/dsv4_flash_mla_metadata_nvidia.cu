#include "../../../handle.h"
#include "dsv4_flash_mla_metadata_nvidia.cuh"

#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <tvm/ffi/container/tensor.h>

namespace {
using flash_mla_metadata_fn_t = void (*)(
    tvm::ffi::TensorView,
    long,
    long,
    tvm::ffi::TensorView,
    tvm::ffi::TensorView);

constexpr const char *kFlashMlaMetadataSymbol = "_Z49flash_mla_get_mla_decoding_metadata_dense_fp8_outN3tvm3ffi10TensorViewEllS1_S1_";

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

flash_mla_metadata_fn_t resolveFlashMlaMetadata() {
    static flash_mla_metadata_fn_t fn = []() -> flash_mla_metadata_fn_t {
        tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
        tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL);
        void *handle = tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            return nullptr;
        }
        return reinterpret_cast<flash_mla_metadata_fn_t>(dlsym(handle, kFlashMlaMetadataSymbol));
    }();
    return fn;
}

tvm::ffi::TensorView makeInt32TensorView(void *data,
                                         const std::array<int64_t, 1> &shape,
                                         const std::array<int64_t, 1> &strides,
                                         int device_id,
                                         DLTensor &tensor) {
    tensor.data = data;
    tensor.device = DLDevice{kDLROCM, device_id};
    tensor.ndim = 1;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = const_cast<int64_t *>(shape.data());
    tensor.strides = const_cast<int64_t *>(strides.data());
    tensor.byte_offset = 0;
    return tvm::ffi::TensorView(&tensor);
}

tvm::ffi::TensorView makeInt32TensorView(void *data,
                                         const std::array<int64_t, 2> &shape,
                                         const std::array<int64_t, 2> &strides,
                                         int device_id,
                                         DLTensor &tensor) {
    tensor.data = data;
    tensor.device = DLDevice{kDLROCM, device_id};
    tensor.ndim = 2;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = const_cast<int64_t *>(shape.data());
    tensor.strides = const_cast<int64_t *>(strides.data());
    tensor.byte_offset = 0;
    return tvm::ffi::TensorView(&tensor);
}

} // namespace
#endif

namespace op::dsv4_flash_mla_metadata::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t cache_seqlens_desc, infiniopTensorDescriptor_t tile_scheduler_metadata_desc, infiniopTensorDescriptor_t num_splits_desc, int num_heads_per_head_k, int num_heads_k) {
    Info info;
    CHECK_STATUS(createInfo(&info, cache_seqlens_desc, tile_scheduler_metadata_desc, num_splits_desc, num_heads_per_head_k, num_heads_k));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, const void *cache_seqlens, void *tile_scheduler_metadata, void *num_splits, void *) const {
#if defined(ENABLE_HYGON_API)
    auto fn = resolveFlashMlaMetadata();
    CHECK_OR_RETURN(fn != nullptr, INFINI_STATUS_INTERNAL_ERROR);

    const auto &i = _info;
    int device_id = this->device_id;
    std::array<int64_t, 1> seqlens_shape{static_cast<int64_t>(i.batch)};
    std::array<int64_t, 1> seqlens_strides{1};
    std::array<int64_t, 2> meta_shape{static_cast<int64_t>(i.tile_meta_rows), 8};
    std::array<int64_t, 2> meta_strides{8, 1};
    std::array<int64_t, 1> splits_shape{static_cast<int64_t>(i.batch + 1)};
    std::array<int64_t, 1> splits_strides{1};

    DLTensor seqlens_t, meta_t, splits_t;
    auto seqlens_v = makeInt32TensorView(const_cast<void *>(cache_seqlens), seqlens_shape, seqlens_strides, device_id, seqlens_t);
    auto meta_v = makeInt32TensorView(tile_scheduler_metadata, meta_shape, meta_strides, device_id, meta_t);
    auto splits_v = makeInt32TensorView(num_splits, splits_shape, splits_strides, device_id, splits_t);

    fn(seqlens_v, static_cast<long>(i.num_heads_per_head_k), static_cast<long>(i.num_heads_k), meta_v, splits_v);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}

} // namespace op::dsv4_flash_mla_metadata::nvidia
