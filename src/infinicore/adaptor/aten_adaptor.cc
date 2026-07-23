#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

#if defined(ENABLE_ILUVATAR_API)
extern "C" int32_t torch_set_current_cuda_stream(void *stream, int32_t device_index);
#endif

namespace infinicore::adaptor {
namespace {
struct AtenTensorCacheKey {
    void *data;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int dtype;
    int device_type;
    int device_index;

    bool operator==(const AtenTensorCacheKey &other) const {
        return data == other.data && sizes == other.sizes
            && strides == other.strides && dtype == other.dtype
            && device_type == other.device_type
            && device_index == other.device_index;
    }
};

struct AtenTensorCacheKeyHash {
    size_t operator()(const AtenTensorCacheKey &key) const {
        size_t hash = std::hash<void *>{}(key.data);
        const auto combine = [&hash](size_t value) {
            hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        };
        combine(std::hash<int>{}(key.dtype));
        combine(std::hash<int>{}(key.device_type));
        combine(std::hash<int>{}(key.device_index));
        for (const auto value : key.sizes) {
            combine(std::hash<int64_t>{}(value));
        }
        for (const auto value : key.strides) {
            combine(std::hash<int64_t>{}(value));
        }
        return hash;
    }
};
} // namespace

at::Tensor to_aten_tensor(const infinicore::Tensor &t) {
    void *data_ptr = (void *)(t->data());

    auto sizes = std::vector<int64_t>(
        t->shape().begin(),
        t->shape().end());

    const auto tensor_strides = t->strides();
    auto strides = std::vector<int64_t>(tensor_strides.begin(), tensor_strides.end());

    // vLLM keeps ATen tensors alive across calls. InfiniCore tensors are
    // non-owning from the ATen point of view, so cache wrappers per worker
    // thread and reuse them only when address and complete metadata match.
    const bool cache_wrapper = t->numel() != 0;
    static thread_local std::unordered_map<AtenTensorCacheKey,
                                           at::Tensor,
                                           AtenTensorCacheKeyHash>
        wrapper_cache;
    AtenTensorCacheKey cache_key{
        data_ptr,
        sizes,
        strides,
        static_cast<int>(t->dtype()),
        static_cast<int>(t->device().getType()),
        static_cast<int>(t->device().getIndex()),
    };
    if (cache_wrapper) {
        const auto it = wrapper_cache.find(cache_key);
        if (it != wrapper_cache.end()) {
            return it->second;
        }
    }

    auto dtype = to_at_dtype(t->dtype());
    auto device = to_at_device(t->device());

    auto deleter_ = [](void * /*unused*/) mutable {

    };

    at::TensorOptions options = at::TensorOptions()
                                    .dtype(dtype)
                                    .device(device)
                                    .requires_grad(false);

    if (t->numel() == 0) {
        return at::empty_strided(sizes, strides, options);
    }

    auto result = at::from_blob(
        data_ptr,
        sizes,
        strides,
        deleter_,
        options);
    if (cache_wrapper) {
        wrapper_cache.emplace(std::move(cache_key), result);
    }
    return result;
}

#if defined(ENABLE_HYGON_API)
c10::hip::HIPStream get_hip_stream() {
    return c10::hip::getStreamFromExternal(
        hipStream_t(infinicore::context::getStream()), infinicore::context::getDevice().getIndex());
}
#elif defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
c10::cuda::CUDAStream get_cuda_stream() {
    return c10::cuda::getStreamFromExternal(
        cudaStream_t(infinicore::context::getStream()), infinicore::context::getDevice().getIndex());
}
#endif

#if defined(ENABLE_ILUVATAR_API)
void set_aten_stream_to_infinicore() {
    const auto error = torch_set_current_cuda_stream(
        infinicore::context::getStream(),
        static_cast<int32_t>(infinicore::context::getDevice().getIndex()));
    if (error != 0) {
        throw std::runtime_error(
            "torch_set_current_cuda_stream failed with error code "
            + std::to_string(error));
    }
}
#endif

#if defined(ENABLE_MOORE_API)
c10::musa::MUSAStream get_musa_stream() {
    return c10::musa::getStreamFromExternal(
        musaStream_t(infinicore::context::getStream()), infinicore::context::getDevice().getIndex());
}
#endif

} // namespace infinicore::adaptor

#endif // ENABLE_ATEN
