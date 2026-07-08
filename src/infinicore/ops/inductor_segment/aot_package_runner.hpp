#pragma once

#ifdef ENABLE_ATEN

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <memory>
#include <string>
#include <vector>

namespace infinicore::op::inductor_segment_impl {

/// Thin wrapper around ``AOTIModelPackageLoader`` for piecewise segments.
class AotPackageRunner {
public:
    explicit AotPackageRunner(const std::string &package_path);

    std::vector<at::Tensor> run(
        const std::vector<at::Tensor> &inputs,
        void *stream_handle = nullptr);

    void warmup(const std::vector<at::Tensor> &inputs);

private:
    void ensure_loader_();

    std::string package_path_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
};

} // namespace infinicore::op::inductor_segment_impl

#endif // ENABLE_ATEN
