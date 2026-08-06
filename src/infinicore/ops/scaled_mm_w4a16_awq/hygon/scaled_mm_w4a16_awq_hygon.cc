#if defined(ENABLE_HYGON_API) && defined(ENABLE_VENDOR_OPS)

#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/scaled_mm_w4a16_awq.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace infinicore::op::scaled_mm_w4a16_awq_impl::hygon {
namespace {

struct __attribute__((packed)) AwqGemmArgs {
    uint32_t gemm_count;
    uint32_t internal_args;
    uint32_t internal_args_1;
    uint32_t num_work_groups;
    uint32_t m;
    uint32_t n;
    uint32_t batch;
    uint32_t k;
    void *d;
    void *c;
    const void *a;
    const void *b;
    uint32_t stride_d_1;
    uint32_t stride_d_2;
    uint32_t stride_c_1;
    uint32_t stride_c_2;
    uint32_t stride_a_1;
    uint32_t stride_a_2;
    uint32_t stride_b_1;
    uint32_t stride_b_2;
    float alpha;
    float beta;
    void *debug_buffer;
    const void *dst_d;
    const void *synchronizer;
    uint32_t gsu_sync;
};

struct KernelConfig {
    const char *kernel_name;
    const char *co_file;
    uint32_t mt_m;
    uint32_t mt_n;
    uint32_t threads;
};

constexpr KernelConfig DECODE_SMALL{
    "Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT16x32x32_SN_K1_PGR6_SB1_TT2_2_WG16_16_2",
    "Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT16x32x32_SN_K1_PGR6_SB1_TT4_2_w4a16.co",
    16, 32, 512};
constexpr KernelConfig DECODE_WIDE{
    "Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT64x32x32_SN_K1_PGR6_SB1_TT2_2_WG16_16_3",
    "Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT64x32x32_SN_K1_PGR6_SB1_TT4_2_w4a16_splitK.co",
    64, 32, 768};
constexpr KernelConfig PREFILL{
    "Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT64x128x32_SN_K1_PGR6_SB1_TT2_8_WG16_16_2",
    "Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT64x128x32_SN_K1_PGR6_SB1_TT4_2_w4a16.co",
    64, 128, 512};

void check_hip(hipError_t status, const char *operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: "
                                 + hipGetErrorString(status));
    }
}

std::string asm_directory() {
    if (const char *configured = std::getenv("INFINICORE_HYGON_AITER_ASM_DIR")) {
        if (*configured != '\0') {
            return configured;
        }
    }
    if (const char *configured = std::getenv("AITER_ASM_DIR")) {
        if (*configured != '\0') {
            return configured;
        }
    }
    return "/usr/local/lib/python3.10/dist-packages/aiter_meta/hsa/gfx936";
}

class Kernel {
public:
    explicit Kernel(const KernelConfig &config)
        : config_(config) {
        auto path = asm_directory();
        if (!path.empty() && path.back() != '/') {
            path.push_back('/');
        }
        path += config.co_file;
        check_hip(hipModuleLoad(&module_, path.c_str()), "hipModuleLoad");
        check_hip(hipModuleGetFunction(&function_, module_, config.kernel_name),
                  "hipModuleGetFunction");
    }

    ~Kernel() {
        if (module_ != nullptr) {
            hipModuleUnload(module_);
        }
    }

    void launch(Tensor out, const Tensor &input, const Tensor &qweight,
                const Tensor &qzeros, const Tensor &scales) const {
        AwqGemmArgs args{};
        args.gemm_count = 1;
        args.internal_args = 0x00200001;
        args.internal_args_1 = 1;
        args.batch = 1;
        // Preserve the vendor's packed-A ABI: m is packed N/2 and n is the
        // activation row count, rather than the conventional GEMM M/N pair.
        args.m = static_cast<uint32_t>(qweight->size(1));
        args.n = static_cast<uint32_t>(input->size(0));
        args.k = static_cast<uint32_t>(input->size(1));
        const uint32_t wg_m = (args.m + config_.mt_m - 1) / config_.mt_m;
        const uint32_t wg_n = (args.n + config_.mt_n - 1) / config_.mt_n;
        args.num_work_groups = wg_m * wg_n;
        args.d = out->data();
        args.c = out->data();
        args.a = qweight->data();
        args.b = input->data();
        const auto output_columns = static_cast<uint32_t>(out->size(1));
        args.stride_d_1 = output_columns;
        args.stride_d_2 = output_columns;
        args.stride_c_1 = output_columns;
        args.stride_c_2 = output_columns;
        args.stride_a_1 = static_cast<uint32_t>(qweight->size(1));
        args.stride_a_2 = static_cast<uint32_t>(qweight->size(1));
        args.stride_b_1 = args.k;
        args.stride_b_2 = args.k;
        args.alpha = 1.0f;
        args.beta = 0.0f;
        args.dst_d = qzeros->data();
        args.synchronizer = scales->data();

        // The vendor kernel metadata declares 132 bytes. Some host compilers
        // retain tail padding, so do not expose sizeof(AwqGemmArgs) here.
        size_t args_size = offsetof(AwqGemmArgs, gsu_sync) + sizeof(args.gsu_sync);
        void *launch_config[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            &args,
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            &args_size,
            HIP_LAUNCH_PARAM_END,
        };
        auto stream = reinterpret_cast<hipStream_t>(context::getStream());
        check_hip(
            hipModuleLaunchKernel(function_, args.num_work_groups, 1, 1,
                                  config_.threads, 1, 1, 0, stream, nullptr,
                                  launch_config),
            "hipModuleLaunchKernel");
    }

private:
    KernelConfig config_;
    hipModule_t module_ = nullptr;
    hipFunction_t function_ = nullptr;
};

const KernelConfig &select_config(size_t m, size_t n) {
    if (m <= 16) {
        return n <= 3072 ? DECODE_SMALL : DECODE_WIDE;
    }
    return PREFILL;
}

Kernel &get_kernel(const KernelConfig &config) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = kernels.find(config.kernel_name);
    if (it == kernels.end()) {
        it = kernels.emplace(config.kernel_name,
                             std::make_unique<Kernel>(config))
                 .first;
    }
    return *it->second;
}

} // namespace

void run(Tensor out, const Tensor &input, const Tensor &qweight,
         const Tensor &qzeros, const Tensor &scales,
         std::optional<Tensor> bias) {
    if (bias) {
        throw std::runtime_error(
            "Hygon scaled_mm_w4a16_awq does not support bias yet");
    }
    const auto &config = select_config(input->size(0), out->size(1));
    get_kernel(config).launch(out, input, qweight, qzeros, scales);
}

static bool registered = []() {
    vendor_ops::scaled_mm_w4a16_awq_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::scaled_mm_w4a16_awq_impl::hygon

#endif
