#if defined(ENABLE_HYGON_API) && defined(ENABLE_VENDOR_OPS) && defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/ops/moe_w4a8_marlin.hpp"

#include <ATen/ATen.h>
#include <c10/hip/HIPGuard.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op::moe_w4a8_marlin_impl::hygon {
namespace {

using VendorMoeW4A8 = at::Tensor (*)(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    std::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor,
    int64_t, int, int);

void initialize_lightop_environment(const char *library_path) {
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess
        || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        throw std::runtime_error(
            "failed to query the Hygon device for the W4A8 vendor kernel");
    }

    std::string architecture = properties.gcnArchName;
    const auto feature_separator = architecture.find(':');
    if (feature_separator != std::string::npos) {
        architecture.resize(feature_separator);
    }
    if (architecture.empty()) {
        throw std::runtime_error(
            "the Hygon device did not report a GPU architecture");
    }

    if (std::getenv("LIGHTOP_GPU_TARGET") == nullptr) {
        setenv("LIGHTOP_GPU_TARGET", architecture.c_str(), 0);
    }
    if (std::getenv("LIGHTOP_GPU_CUS") == nullptr) {
        const auto compute_units = std::to_string(properties.multiProcessorCount);
        setenv("LIGHTOP_GPU_CUS", compute_units.c_str(), 0);
    }
    if (std::getenv("LIGHTOP_ASM_DIR") == nullptr) {
        std::string root = library_path;
        const auto separator = root.rfind('/');
        if (separator == std::string::npos) {
            throw std::runtime_error(
                "cannot infer the LightOp root from the vendor library path");
        }
        root.resize(separator);
        const auto assembly_dir = root + "/hsa/" + architecture + "/";
        setenv("LIGHTOP_ASM_DIR", assembly_dir.c_str(), 0);
    }
}

VendorMoeW4A8 vendor_moe_gemm_w4a8() {
    static const auto function = [] {
        const char *configured = std::getenv("INFINICORE_HYGON_LIGHTOP_SO");
        const char *path = configured != nullptr && configured[0] != '\0'
                             ? configured
                             : "/usr/local/lib/python3.10/dist-packages/"
                               "lightop/op.cpython-310-x86_64-linux-gnu.so";
        initialize_lightop_environment(path);

        constexpr const char *symbol = "_ZN2at6native13moe_gemm_w4a8ENS_6TensorES1_S1_S1_S1_"
                                       "St8optionalIS1_ES1_S1_S1_lii";
        dlerror();
        if (void *address = dlsym(RTLD_DEFAULT, symbol)) {
            return reinterpret_cast<VendorMoeW4A8>(address);
        }
        dlerror();
        void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (handle == nullptr) {
            const char *error = dlerror();
            throw std::runtime_error(
                std::string("failed to load Hygon W4A8 vendor library: ")
                + (error != nullptr ? error : "unknown loader error"));
        }
        dlerror();
        void *address = dlsym(handle, symbol);
        if (address == nullptr) {
            const char *error = dlerror();
            throw std::runtime_error(
                std::string("Hygon W4A8 vendor symbol is unavailable: ")
                + (error != nullptr ? error : "unknown loader error"));
        }
        return reinterpret_cast<VendorMoeW4A8>(address);
    }();
    return function;
}

} // namespace

void run_vendor(Tensor output,
                const Tensor &input,
                const Tensor &marlin_weight,
                const Tensor &input_scale,
                const Tensor &weight_scale,
                std::optional<Tensor> topk_weights,
                const Tensor &padded_sorted_token_ids,
                const Tensor &expert_ids,
                const Tensor &num_tokens_post_pad,
                int64_t topk) {
    const auto device_index = static_cast<c10::DeviceIndex>(input->device().getIndex());
    const auto set_device_status = hipSetDevice(device_index);
    if (set_device_status != hipSuccess) {
        throw std::runtime_error(
            "failed to select the Hygon W4A8 vendor device: "
            + std::string(hipGetErrorString(set_device_status)));
    }
    const auto stream = adaptor::get_hip_stream();
    c10::hip::HIPStreamGuard stream_guard(stream);

    const auto experts = static_cast<int64_t>(marlin_weight->size(0));
    const auto k_blocks = static_cast<int64_t>(input->size(1) / 32);
    const auto packed_columns = static_cast<int64_t>(output->size(1) * 4);
    auto weight_at = adaptor::to_aten_tensor(marlin_weight)
                         .view(at::kInt)
                         .view({experts, k_blocks, packed_columns});
    std::optional<at::Tensor> topk_weights_at = std::nullopt;
    if (topk_weights) {
        topk_weights_at = adaptor::to_aten_tensor(*topk_weights);
    }
    const bool first_stage = output->size(1) <= 512;
    const int mode = input->size(0) == 2 ? 402
                                         : (first_stage ? 403 : 401);
    vendor_moe_gemm_w4a8()(
        adaptor::to_aten_tensor(input), weight_at,
        adaptor::to_aten_tensor(output), adaptor::to_aten_tensor(input_scale),
        adaptor::to_aten_tensor(weight_scale), topk_weights_at,
        adaptor::to_aten_tensor(padded_sorted_token_ids),
        adaptor::to_aten_tensor(expert_ids),
        adaptor::to_aten_tensor(num_tokens_post_pad),
        topk, mode, 1);
}

} // namespace infinicore::op::moe_w4a8_marlin_impl::hygon
#endif
