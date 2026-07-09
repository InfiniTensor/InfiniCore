#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/adaptor/lightop_adaptor.hpp"

#include <dlfcn.h>

#include <cctype>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace infinicore::adaptor::lightop {
namespace {

constexpr const char *kDefaultLightopSo =
    "/usr/local/lib/python3.10/dist-packages/lightop/op.cpython-310-x86_64-linux-gnu.so";

constexpr const char *kFusedRmsNormSymbol =
    "_ZN2at6native25fused_rms_norm_contiguousERNS_6TensorES2_S2_d";
constexpr const char *kFuseSiluAndMulSymbol =
    "_ZN2at6native17fuse_silu_and_mulERNS_6TensorES2_";
constexpr const char *kMoeSumSymbol =
    "_ZN2at6native7moe_sumERNS_6TensorES2_RKSt8optionalIS1_ES6_S6_fi";
constexpr const char *kMoeFusedGateSymbol =
    "_ZN2at6native14moe_fused_gateERNS_6TensorES2_lllld";
constexpr const char *kMoeGemmW16A16Symbol =
    "_ZN2at6native15moe_gemm_w16a16ENS_6TensorES1_S1_St8optionalIS1_ES1_S1_S1_lii";
constexpr const char *kMoeMarlinW16A16AsmSymbol =
    "_ZN2at6native21moe_marlin_w16a16_asmENS_6TensorES1_S1_St8optionalIS1_ES1_S1_S1_iii";

class LightopLibrary {
public:
    bool available() {
        std::lock_guard<std::mutex> lock(mutex_);
        return ensure_open_locked(false);
    }

    void *symbol(const char *name) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ensure_open_locked(true)) {
            throw std::runtime_error(error_);
        }

        dlerror();
        void *fn = dlsym(handle_, name);
        const char *err = dlerror();
        if (err != nullptr || fn == nullptr) {
            std::ostringstream oss;
            oss << "failed to resolve lightop symbol " << name;
            if (err != nullptr) {
                oss << ": " << err;
            }
            throw std::runtime_error(oss.str());
        }
        return fn;
    }

private:
    bool ensure_open_locked(bool update_error) {
        if (handle_ != nullptr) {
            return true;
        }

        const char *path_env = std::getenv("INFINICORE_LIGHTOP_SO");
        const char *path = (path_env != nullptr && path_env[0] != '\0') ? path_env : kDefaultLightopSo;
        handle_ = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
        if (handle_ != nullptr) {
            error_.clear();
            return true;
        }

        if (update_error || error_.empty()) {
            const char *err = dlerror();
            std::ostringstream oss;
            oss << "failed to load lightop shared library " << path;
            if (err != nullptr) {
                oss << ": " << err;
            }
            error_ = oss.str();
        }
        return false;
    }

    std::mutex mutex_;
    void *handle_ = nullptr;
    std::string error_;
};

LightopLibrary &library() {
    static LightopLibrary lib;
    return lib;
}

template <typename Fn>
Fn resolve(const char *symbol) {
    return reinterpret_cast<Fn>(library().symbol(symbol));
}

using FusedRmsNormFn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, double);
using FuseSiluAndMulFn = void (*)(at::Tensor &, at::Tensor &);
using MoeSumFn = void (*)(
    at::Tensor &,
    at::Tensor &,
    const std::optional<at::Tensor> &,
    const std::optional<at::Tensor> &,
    const std::optional<at::Tensor> &,
    float,
    int);
using MoeFusedGateFn = std::vector<at::Tensor> (*)(
    at::Tensor &,
    at::Tensor &,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    double);
using MoeGemmW16A16Fn = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    int64_t,
    int,
    int);
using MoeMarlinW16A16AsmFn = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    int,
    int,
    int);

FusedRmsNormFn fused_rms_norm_fn() {
    static auto fn = resolve<FusedRmsNormFn>(kFusedRmsNormSymbol);
    return fn;
}

FuseSiluAndMulFn fuse_silu_and_mul_fn() {
    static auto fn = resolve<FuseSiluAndMulFn>(kFuseSiluAndMulSymbol);
    return fn;
}

MoeSumFn moe_sum_fn() {
    static auto fn = resolve<MoeSumFn>(kMoeSumSymbol);
    return fn;
}

MoeFusedGateFn moe_fused_gate_fn() {
    static auto fn = resolve<MoeFusedGateFn>(kMoeFusedGateSymbol);
    return fn;
}

MoeGemmW16A16Fn moe_gemm_w16a16_fn() {
    static auto fn = resolve<MoeGemmW16A16Fn>(kMoeGemmW16A16Symbol);
    return fn;
}

MoeMarlinW16A16AsmFn moe_marlin_w16a16_asm_fn() {
    static auto fn = resolve<MoeMarlinW16A16AsmFn>(kMoeMarlinW16A16AsmSymbol);
    return fn;
}

} // namespace

bool available() {
    return library().available();
}

bool enabled_by_env() {
    const char *value = std::getenv("INFINICORE_ENABLE_HYGON_LIGHTOP");
    if (value == nullptr) {
        return false;
    }
    std::string normalized(value);
    for (auto &ch : normalized) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes";
}

void preload_basic_ops() {
    (void)fused_rms_norm_fn();
    (void)fuse_silu_and_mul_fn();
    (void)moe_sum_fn();
    (void)moe_fused_gate_fn();
    (void)moe_gemm_w16a16_fn();
    (void)moe_marlin_w16a16_asm_fn();
}

void preload_silu_and_mul() {
    (void)fuse_silu_and_mul_fn();
}

void fused_rms_norm_contiguous(at::Tensor &out, at::Tensor &input, at::Tensor &weight, double epsilon) {
    fused_rms_norm_fn()(out, input, weight, epsilon);
}

void fuse_silu_and_mul(at::Tensor &input, at::Tensor &output) {
    fuse_silu_and_mul_fn()(input, output);
}

void moe_sum(at::Tensor &input,
             at::Tensor &output,
             const std::optional<at::Tensor> &bias,
             const std::optional<at::Tensor> &expert_mask,
             const std::optional<at::Tensor> &local_num_tokens,
             float factor,
             int expect_m) {
    moe_sum_fn()(input, output, bias, expert_mask, local_num_tokens, factor, expect_m);
}

std::vector<at::Tensor> moe_fused_gate(at::Tensor &input,
                                       at::Tensor &bias,
                                       int64_t num_expert_group,
                                       int64_t topk_group,
                                       int64_t topk,
                                       int64_t num_fused_shared_experts,
                                       double routed_scaling_factor) {
    return moe_fused_gate_fn()(input, bias, num_expert_group, topk_group, topk, num_fused_shared_experts, routed_scaling_factor);
}

void moe_gemm_marlin_w16a16(at::Tensor input,
                            at::Tensor b_qweight,
                            at::Tensor output,
                            const std::optional<at::Tensor> &topk_weights,
                            at::Tensor sorted_token_ids,
                            at::Tensor expert_ids,
                            at::Tensor num_tokens_post_padded,
                            int64_t top_k,
                            int mode,
                            int delta) {
    if (mode < 1000) {
        moe_gemm_w16a16_fn()(
            input, b_qweight, output, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            top_k, mode, delta);
    } else {
        moe_marlin_w16a16_asm_fn()(
            input, b_qweight, output, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            static_cast<int>(top_k), mode, delta);
    }
}

} // namespace infinicore::adaptor::lightop

#endif // ENABLE_HYGON_API && ENABLE_ATEN
