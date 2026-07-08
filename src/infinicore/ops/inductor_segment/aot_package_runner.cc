#ifdef ENABLE_ATEN

#include "aot_package_runner.hpp"
#include "inductor_segment_registry.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"

#include <stdexcept>

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif

namespace infinicore::op::inductor_segment_impl {

AotPackageRunner::AotPackageRunner(const std::string &package_path)
    : package_path_(package_path) {}

void AotPackageRunner::ensure_loader_() {
    if (loader_) {
        return;
    }
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    const int device_index = static_cast<int>(infinicore::context::getDevice().getIndex());
    c10::cuda::CUDAGuard device_guard(device_index);
#endif
    loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        package_path_,
        "model",
        /*run_single_threaded=*/true);
}

std::vector<at::Tensor> AotPackageRunner::run(
    const std::vector<at::Tensor> &inputs,
    void *stream_handle) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    const int rank_device_index = static_cast<int>(infinicore::context::getDevice().getIndex());
    c10::cuda::CUDAGuard rank_guard(rank_device_index);
#endif
    ensure_loader_();
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    void *stream = stream_handle != nullptr
                       ? stream_handle
                       : c10::cuda::getCurrentCUDAStream().stream();
#else
    void *stream = stream_handle != nullptr
                       ? stream_handle
                       : infinicore::context::getStream();
#endif
    auto outputs = loader_->run(inputs, stream);
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    c10::cuda::CUDAGuard restore_guard(rank_device_index);
#endif
    return outputs;
}

void AotPackageRunner::warmup(const std::vector<at::Tensor> &inputs) {
    (void)run(inputs, nullptr);
}

} // namespace infinicore::op::inductor_segment_impl

#endif // ENABLE_ATEN
