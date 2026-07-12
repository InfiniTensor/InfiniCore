#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/adaptor/lightop_adaptor.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace infinicore::adaptor::lightop {
namespace {

constexpr const char *kDefaultLightopSo =
    "/usr/local/lib/python3.10/dist-packages/lightop/op.cpython-310-x86_64-linux-gnu.so";
constexpr const char *kDefaultLmslimQuantSo =
    "/usr/local/lib/python3.10/dist-packages/lmslimquant.cpython-310-x86_64-linux-gnu.so";
constexpr const char *kDefaultLightopGpuTarget = "gfx936";
constexpr const char *kDefaultLightopAsmDir =
    "/usr/local/lib/python3.10/dist-packages/lightop/hsa/gfx936/";
constexpr const char *kFuseSiluAndMulSymbol =
    "_ZN2at6native17fuse_silu_and_mulERNS_6TensorES2_";
constexpr const char *kRmsRotaryEmbeddingFuseSymbol =
    "_ZN2at6native25rms_rotary_embedding_fuseERNS_6TensorES2_S2_lS2_bS1_S1_St8optionalIS1_ES4_d";
constexpr const char *kMoeSumSymbol =
    "_ZN2at6native7moe_sumERNS_6TensorES2_RKSt8optionalIS1_ES6_S6_fi";
constexpr const char *kMoeAlignBlockSizeSymbol =
    "_ZN2at6native20moe_align_block_sizeENS_6TensorEllS1_S1_S1_RKSt8optionalIS1_ES5_S5_bb";
constexpr const char *kMoeGemmW16A16Symbol =
    "_ZN2at6native15moe_gemm_w16a16ENS_6TensorES1_S1_St8optionalIS1_ES1_S1_S1_lii";
constexpr const char *kMoeMarlinW16A16AsmSymbol =
    "_ZN2at6native21moe_marlin_w16a16_asmENS_6TensorES1_S1_St8optionalIS1_ES1_S1_S1_iii";
constexpr const char *kMoeGemmW8A8Symbol =
    "_ZN2at6native20moe_gemm_marlin_w8a8ENS_6TensorES1_S1_S1_S1_St8optionalIS1_ES1_S1_S1_lii";
constexpr const char *kMoeMarlinW8A8AsmSymbol =
    "_ZN2at6native19moe_marlin_w8a8_asmENS_6TensorES1_S1_S1_S1_St8optionalIS1_ES1_S1_S1_jii";
constexpr const char *kFuseSiluMulQuantSymbol =
    "_ZN2at6native19fuse_silu_mul_quantERNS_6TensorES2_S2_RSt8optionalIS1_EiiS5_";
constexpr const char *kPerTokenDynamicQuantInt8Symbol =
    "_ZN2at6native28per_token_dynamic_quant_int8ERNS_6TensorERKS1_S2_S4_";
constexpr const char *kBlasltW8A8Bf16Symbol =
    "_ZN14hipblaslt_gemm14w8a8_bf16_gemmERKN2at6TensorES3_S3_S3_RS1_llllRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES3_S3_RKSt8optionalIS1_E";
constexpr const char *kBlasltW8A8Fp16Symbol =
    "_ZN14hipblaslt_gemm14w8a8_fp16_gemmERKN2at6TensorES3_S3_S3_RS1_llllRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES3_S3_RKSt8optionalIS1_E";

void ensure_default_lightop_env();

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

        ensure_default_lightop_env();

        const char *path_env = std::getenv("INFINICORE_LIGHTOP_SO");
        const char *path = (path_env != nullptr && path_env[0] != '\0') ? path_env : kDefaultLightopSo;
        handle_ = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
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

constexpr const char *kMoeW8A8MarlinMode1001Co =
    "moe_w8a8_channel/moe_w8a8_i8_marlin_64x256x128_TN_BF16_UP.co";
constexpr const char *kMoeW8A8MarlinMode1001Kernel =
    "MOE_W8A8_I8_PERCHANNEL_MARLIN_ASM_TN_MT64x256x128_WGM1_UP";
constexpr uint32_t kMoeW8A8MarlinNBlock = 256;
constexpr uint32_t kMoeW8A8MarlinWorkgroupSize = 768;

struct MoeW8A8MarlinMode1001Args {
    uint32_t n_block_count;
    uint32_t max_m_block_count;
    void *output;
    void *weight;
    void *input;
    void *weight_scale;
    void *input_scale;
    void *topk_weights;
    void *sorted_token_ids;
    void *expert_ids;
    void *num_tokens_post_padded;
    uint32_t num_experts;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t flag0;
    uint32_t flag1;
    uint32_t output_stride;
    uint32_t flag2;
    uint32_t flag3;
    uint32_t max_tokens_padded;
    uint32_t top_k;
    float inverse_top_k;
    float output_scale;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};
static_assert(sizeof(MoeW8A8MarlinMode1001Args) == 144);

struct MoeW8A8MarlinDeviceKernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
};

std::mutex &moe_w8a8_marlin_kernel_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<int, MoeW8A8MarlinDeviceKernel> &moe_w8a8_marlin_kernels() {
    static std::unordered_map<int, MoeW8A8MarlinDeviceKernel> kernels;
    return kernels;
}

std::string hip_error_message(const std::string &operation, hipError_t status) {
    std::ostringstream oss;
    oss << operation << " failed with HIP status " << static_cast<int>(status);
    const char *message = hipGetErrorString(status);
    if (message != nullptr) {
        oss << " (" << message << ")";
    }
    return oss.str();
}

MoeW8A8MarlinDeviceKernel get_moe_w8a8_marlin_mode1001_kernel() {
    int device = -1;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipGetDevice", status));
    }

    std::lock_guard<std::mutex> lock(moe_w8a8_marlin_kernel_mutex());
    auto &kernels = moe_w8a8_marlin_kernels();
    auto found = kernels.find(device);
    if (found != kernels.end()) {
        return found->second;
    }

    ensure_default_lightop_env();
    const char *asm_dir_env = std::getenv("LIGHTOP_ASM_DIR");
    std::string asm_dir =
        asm_dir_env != nullptr && asm_dir_env[0] != '\0'
            ? asm_dir_env
            : kDefaultLightopAsmDir;
    if (!asm_dir.empty() && asm_dir.back() != '/') {
        asm_dir.push_back('/');
    }
    const std::string co_path = asm_dir + kMoeW8A8MarlinMode1001Co;

    MoeW8A8MarlinDeviceKernel kernel;
    status = hipModuleLoad(&kernel.module, co_path.c_str());
    if (status != hipSuccess) {
        throw std::runtime_error(hip_error_message("hipModuleLoad(" + co_path + ")", status));
    }
    status = hipModuleGetFunction(
        &kernel.function, kernel.module, kMoeW8A8MarlinMode1001Kernel);
    if (status != hipSuccess) {
        (void)hipModuleUnload(kernel.module);
        throw std::runtime_error(hip_error_message("hipModuleGetFunction", status));
    }

    kernels.emplace(device, kernel);
    return kernel;
}

uint32_t checked_u32(int64_t value, const char *name) {
    if (value < 0 ||
        static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("Hygon W8A8 Marlin ") + name + " exceeds uint32");
    }
    return static_cast<uint32_t>(value);
}

bool can_launch_moe_w8a8_marlin_mode1001(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &output,
    const at::Tensor &input_scale,
    const at::Tensor &weight_scale,
    const std::optional<at::Tensor> &topk_weights,
    const at::Tensor &sorted_token_ids,
    const at::Tensor &expert_ids,
    const at::Tensor &num_tokens_post_padded,
    int64_t top_k,
    int mode,
    int delta) {
    return mode == 1001 && delta == 1 && !topk_weights.has_value() &&
           top_k > 0 &&
           input.dim() == 2 && weight.dim() == 3 && output.dim() == 3 &&
           input_scale.dim() == 2 && weight_scale.dim() == 3 &&
           input.scalar_type() == at::kChar &&
           weight.scalar_type() == at::kChar &&
           output.scalar_type() == at::kBFloat16 &&
           input_scale.scalar_type() == at::kFloat &&
           weight_scale.scalar_type() == at::kFloat &&
           sorted_token_ids.scalar_type() == at::kInt &&
           expert_ids.scalar_type() == at::kInt &&
           num_tokens_post_padded.scalar_type() == at::kInt &&
           input.is_contiguous() && weight.is_contiguous() &&
           output.is_contiguous() && input_scale.is_contiguous() &&
           weight_scale.is_contiguous() && sorted_token_ids.is_contiguous() &&
           expert_ids.is_contiguous() && num_tokens_post_padded.is_contiguous();
}

void launch_moe_w8a8_marlin_mode1001(
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &output,
    at::Tensor &input_scale,
    at::Tensor &weight_scale,
    at::Tensor &sorted_token_ids,
    at::Tensor &expert_ids,
    at::Tensor &num_tokens_post_padded,
    int64_t top_k) {
    const int64_t m = input.size(0);
    const int64_t k = input.size(1);
    const int64_t num_experts = weight.size(0);
    const int64_t n = output.size(2);
    if (output.size(0) != m || output.size(1) != top_k ||
        weight.size(1) * 64 != k || weight.size(2) != n * 64 ||
        input_scale.size(0) != m || input_scale.size(1) != 1 ||
        weight_scale.size(0) != num_experts ||
        weight_scale.size(1) != n || weight_scale.size(2) != 1 ||
        num_tokens_post_padded.numel() != 1) {
        throw std::runtime_error("Hygon W8A8 Marlin mode 1001 tensor shape mismatch");
    }

    const uint32_t n_u32 = checked_u32(n, "N");
    const uint32_t top_k_u32 = checked_u32(top_k, "top_k");
    const uint32_t n_block_count =
        (n_u32 + kMoeW8A8MarlinNBlock - 1) / kMoeW8A8MarlinNBlock;
    const uint32_t max_m_block_count =
        checked_u32(expert_ids.numel(), "max_m_block_count");

    MoeW8A8MarlinMode1001Args args{
        n_block_count,
        max_m_block_count,
        output.data_ptr(),
        weight.data_ptr(),
        input.data_ptr(),
        weight_scale.data_ptr(),
        input_scale.data_ptr(),
        nullptr,
        sorted_token_ids.data_ptr(),
        expert_ids.data_ptr(),
        num_tokens_post_padded.data_ptr(),
        checked_u32(num_experts, "num_experts"),
        checked_u32(m, "M"),
        n_u32,
        checked_u32(k, "K"),
        1,
        1,
        n_u32,
        1,
        1,
        checked_u32(sorted_token_ids.numel(), "max_tokens_padded"),
        top_k_u32,
        1.0f / static_cast<float>(top_k_u32),
        1.0f,
        0,
        0,
        0};

    size_t args_size = sizeof(args);
    void *launch_config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &args_size,
        HIP_LAUNCH_PARAM_END};

    auto kernel = get_moe_w8a8_marlin_mode1001_kernel();
    auto status = hipModuleLaunchKernel(
        kernel.function,
        n_block_count,
        1,
        max_m_block_count,
        kMoeW8A8MarlinWorkgroupSize,
        1,
        1,
        0,
        infinicore::adaptor::get_hip_stream().stream(),
        nullptr,
        launch_config);
    if (status != hipSuccess) {
        throw std::runtime_error(
            hip_error_message("hipModuleLaunchKernel(W8A8 Marlin mode 1001)", status));
    }
}

class LmslimQuantLibrary {
public:
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
            oss << "failed to resolve lmslimquant symbol " << name;
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

        const char *path_env = std::getenv("INFINICORE_LMSLIMQUANT_SO");
        const char *path = (path_env != nullptr && path_env[0] != '\0') ? path_env : kDefaultLmslimQuantSo;
        handle_ = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
        if (handle_ != nullptr) {
            error_.clear();
            return true;
        }

        if (update_error || error_.empty()) {
            const char *err = dlerror();
            std::ostringstream oss;
            oss << "failed to load lmslimquant shared library " << path;
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

LmslimQuantLibrary &lmslimquant_library() {
    static LmslimQuantLibrary lib;
    return lib;
}

void ensure_default_lightop_env() {
    if (std::getenv("LIGHTOP_GPU_TARGET") == nullptr) {
        setenv("LIGHTOP_GPU_TARGET", kDefaultLightopGpuTarget, 0);
    }
    if (std::getenv("LIGHTOP_ASM_DIR") == nullptr) {
        setenv("LIGHTOP_ASM_DIR", kDefaultLightopAsmDir, 0);
    }
}

template <typename Fn>
Fn resolve(const char *symbol) {
    return reinterpret_cast<Fn>(library().symbol(symbol));
}

template <typename Fn>
Fn resolve_lmslimquant(const char *symbol) {
    return reinterpret_cast<Fn>(lmslimquant_library().symbol(symbol));
}

using FuseSiluAndMulFn = void (*)(at::Tensor &, at::Tensor &);
using RmsRotaryEmbeddingFuseFn = void (*)(
    at::Tensor &,
    at::Tensor &,
    at::Tensor &,
    long,
    at::Tensor &,
    bool,
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    double);
using MoeSumFn = void (*)(
    at::Tensor &,
    at::Tensor &,
    const std::optional<at::Tensor> &,
    const std::optional<at::Tensor> &,
    const std::optional<at::Tensor> &,
    float,
    int);
using MoeAlignBlockSizeFn = void (*)(
    at::Tensor,
    int64_t,
    int64_t,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    const std::optional<at::Tensor> &,
    const std::optional<at::Tensor> &,
    const std::optional<at::Tensor> &,
    bool,
    bool);
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

using MoeGemmW8A8Fn = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
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
using MoeMarlinW8A8AsmFn = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    unsigned int,
    int,
    int);
using FuseSiluMulQuantFn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, std::optional<at::Tensor> &, int, int, std::optional<at::Tensor> &);
using PerTokenDynamicQuantInt8Fn = void (*)(at::Tensor &, const at::Tensor &, at::Tensor &, const at::Tensor &);
using BlasltW8A8GemmFn = void (*)(
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &,
    long,
    long,
    long,
    long,
    const std::string &,
    const at::Tensor &,
    const at::Tensor &,
    const std::optional<at::Tensor> &);

FuseSiluAndMulFn fuse_silu_and_mul_fn() {
    static auto fn = resolve<FuseSiluAndMulFn>(kFuseSiluAndMulSymbol);
    return fn;
}

RmsRotaryEmbeddingFuseFn rms_rotary_embedding_fuse_fn() {
    static auto fn = resolve<RmsRotaryEmbeddingFuseFn>(kRmsRotaryEmbeddingFuseSymbol);
    return fn;
}

MoeSumFn moe_sum_fn() {
    static auto fn = resolve<MoeSumFn>(kMoeSumSymbol);
    return fn;
}

MoeAlignBlockSizeFn moe_align_block_size_fn() {
    static auto fn = resolve<MoeAlignBlockSizeFn>(kMoeAlignBlockSizeSymbol);
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

MoeGemmW8A8Fn moe_gemm_w8a8_fn() {
    static auto fn = resolve<MoeGemmW8A8Fn>(kMoeGemmW8A8Symbol);
    return fn;
}

MoeMarlinW8A8AsmFn moe_marlin_w8a8_asm_fn() {
    static auto fn = resolve<MoeMarlinW8A8AsmFn>(kMoeMarlinW8A8AsmSymbol);
    return fn;
}

FuseSiluMulQuantFn fuse_silu_mul_quant_fn() {
    static auto fn = resolve<FuseSiluMulQuantFn>(kFuseSiluMulQuantSymbol);
    return fn;
}

PerTokenDynamicQuantInt8Fn per_token_dynamic_quant_int8_fn() {
    static auto fn = resolve<PerTokenDynamicQuantInt8Fn>(kPerTokenDynamicQuantInt8Symbol);
    return fn;
}

BlasltW8A8GemmFn blaslt_w8a8_bf16_fn() {
    static auto fn = resolve_lmslimquant<BlasltW8A8GemmFn>(kBlasltW8A8Bf16Symbol);
    return fn;
}

BlasltW8A8GemmFn blaslt_w8a8_fp16_fn() {
    static auto fn = resolve_lmslimquant<BlasltW8A8GemmFn>(kBlasltW8A8Fp16Symbol);
    return fn;
}

} // namespace

bool available() {
    return library().available();
}

void preload_moe_w16a16_ops() {
    (void)moe_sum_fn();
    (void)moe_gemm_w16a16_fn();
    (void)moe_marlin_w16a16_asm_fn();
}

void preload_moe_w8a8_ops() {
    (void)moe_sum_fn();
    (void)moe_gemm_w8a8_fn();
    (void)moe_marlin_w8a8_asm_fn();
    (void)fuse_silu_mul_quant_fn();
}

void preload_moe_align() {
    (void)moe_align_block_size_fn();
}

void preload_silu_and_mul() {
    (void)fuse_silu_and_mul_fn();
}

void preload_moe_w8a8_marlin_asm() {
    (void)get_moe_w8a8_marlin_mode1001_kernel();
}

void preload_rms_rotary_embedding() {
    (void)rms_rotary_embedding_fuse_fn();
}

void preload_w8a8_linear_ops() {
    (void)per_token_dynamic_quant_int8_fn();
    (void)blaslt_w8a8_bf16_fn();
    (void)blaslt_w8a8_fp16_fn();
}

void fuse_silu_and_mul(at::Tensor &input, at::Tensor &output) {
    fuse_silu_and_mul_fn()(input, output);
}

void rms_rotary_embedding_fuse(at::Tensor &positions,
                               at::Tensor &query,
                               at::Tensor &key,
                               int64_t head_size,
                               at::Tensor &cos_sin_cache,
                               bool is_neox,
                               at::Tensor q_weight,
                               at::Tensor k_weight,
                               const std::optional<at::Tensor> &q_bias,
                               const std::optional<at::Tensor> &k_bias,
                               double epsilon) {
    rms_rotary_embedding_fuse_fn()(
        positions,
        query,
        key,
        static_cast<long>(head_size),
        cos_sin_cache,
        is_neox,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        epsilon);
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

void moe_align_block_size(
    at::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    at::Tensor sorted_token_ids,
    at::Tensor expert_ids,
    at::Tensor num_tokens_post_padded,
    const std::optional<at::Tensor> &expert_map,
    const std::optional<at::Tensor> &expert_mask,
    const std::optional<at::Tensor> &num_local_tokens,
    bool is_ep,
    bool fuse_fill) {
    moe_align_block_size_fn()(
        topk_ids,
        static_cast<long>(num_experts),
        static_cast<long>(block_size),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        expert_map,
        expert_mask,
        num_local_tokens,
        is_ep,
        fuse_fill);
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

void moe_gemm_marlin_w8a8(at::Tensor input,
                          at::Tensor b_qweight,
                          at::Tensor output,
                          at::Tensor a_scale,
                          at::Tensor b_scale,
                          const std::optional<at::Tensor> &topk_weights,
                          at::Tensor sorted_token_ids,
                          at::Tensor expert_ids,
                          at::Tensor num_tokens_post_padded,
                          int64_t top_k,
                          int mode,
                          int delta) {
    if (mode < 1000) {
        moe_gemm_w8a8_fn()(
            input, b_qweight, output, a_scale, b_scale, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            top_k, mode, delta);
    } else {
        if (can_launch_moe_w8a8_marlin_mode1001(
                input, b_qweight, output, a_scale, b_scale, topk_weights,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                top_k, mode, delta)) {
            launch_moe_w8a8_marlin_mode1001(
                input, b_qweight, output, a_scale, b_scale,
                sorted_token_ids, expert_ids, num_tokens_post_padded, top_k);
            return;
        }
        moe_marlin_w8a8_asm_fn()(
            input, b_qweight, output, a_scale, b_scale, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            static_cast<unsigned int>(top_k), mode, delta);
    }
}

void fuse_silu_mul_quant(at::Tensor &input,
                         at::Tensor &output,
                         at::Tensor &scales,
                         std::optional<at::Tensor> &num_local_tokens,
                         int topk,
                         int expect_m,
                         std::optional<at::Tensor> &expert_ids) {
    fuse_silu_mul_quant_fn()(
        input,
        output,
        scales,
        num_local_tokens,
        topk,
        expect_m,
        expert_ids);
}

void per_token_dynamic_quant_int8(at::Tensor &output,
                                  const at::Tensor &input,
                                  at::Tensor &scales,
                                  const at::Tensor &smooth) {
    per_token_dynamic_quant_int8_fn()(output, input, scales, smooth);
}

void blaslt_w8a8_gemm(at::Tensor &output,
                      const at::Tensor &a,
                      const at::Tensor &b,
                      const at::Tensor &scale_a,
                      const at::Tensor &scale_b,
                      const std::optional<at::Tensor> &bias) {
    if (a.dim() != 2 || b.dim() != 2 || output.dim() != 2) {
        throw std::runtime_error("lmslimquant W8A8 GEMM expects 2D tensors");
    }
    if (!a.is_contiguous() || !b.is_contiguous() || !output.is_contiguous()) {
        throw std::runtime_error("lmslimquant W8A8 GEMM expects contiguous a, b, and output");
    }
    if (a.scalar_type() != at::kChar || b.scalar_type() != at::kChar ||
        scale_a.scalar_type() != at::kFloat || scale_b.scalar_type() != at::kFloat) {
        throw std::runtime_error("lmslimquant W8A8 GEMM expects int8 inputs and float32 scales");
    }

    const long m = static_cast<long>(output.size(0));
    const long n = static_cast<long>(output.size(1));
    const long k = static_cast<long>(a.size(1));
    if (a.size(0) != m || b.size(0) != n || b.size(1) != k) {
        throw std::runtime_error("lmslimquant W8A8 GEMM shape mismatch");
    }

    static const at::Tensor alpha = at::tensor(1, at::TensorOptions().dtype(at::kInt));
    static const at::Tensor beta = at::tensor(0, at::TensorOptions().dtype(at::kInt));
    static const std::string transpose = "TN";
    constexpr long batch = 1;

    if (output.scalar_type() == at::kBFloat16) {
        blaslt_w8a8_bf16_fn()(b, a, scale_b, scale_a, output, m, n, k, batch, transpose, alpha, beta, bias);
        return;
    }
    if (output.scalar_type() == at::kHalf) {
        blaslt_w8a8_fp16_fn()(b, a, scale_b, scale_a, output, m, n, k, batch, transpose, alpha, beta, bias);
        return;
    }
    throw std::runtime_error("lmslimquant W8A8 GEMM only supports FP16/BF16 output");
}

} // namespace infinicore::adaptor::lightop

#endif // ENABLE_HYGON_API && ENABLE_ATEN
