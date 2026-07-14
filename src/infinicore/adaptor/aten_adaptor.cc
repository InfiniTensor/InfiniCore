#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <cstdint>

#if defined(ENABLE_ILUVATAR_API)
extern "C" int32_t torch_set_current_cuda_stream(void *stream, int32_t device_index);
#endif

namespace infinicore::adaptor {

at::Tensor to_aten_tensor(const infinicore::Tensor &t) {
    void *data_ptr = (void *)(t->data());

    auto sizes = std::vector<int64_t>(
        t->shape().begin(),
        t->shape().end());

    auto strides = t->strides();

    auto dtype = to_at_dtype(t->dtype());
    auto device = to_at_device(t->device());

    auto deleter_ = [](void * /*unused*/) mutable {

    };

    at::TensorOptions options = at::TensorOptions()
                                    .dtype(dtype)
                                    .device(device)
                                    .requires_grad(false);

    return at::from_blob(
        data_ptr,
        sizes,
        strides,
        deleter_,
        options);
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
