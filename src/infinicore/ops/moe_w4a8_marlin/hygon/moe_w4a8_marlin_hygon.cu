#if defined(ENABLE_HYGON_API) && defined(ENABLE_VENDOR_OPS)
#include "infinicore/context/context.hpp"
#include "infinicore/ops/moe_w4a8_marlin.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

using HygonModule = void *;
using HygonFunction = void *;
using HygonStream = void *;
using HygonEvent = void *;

extern "C" {
int hipModuleLoad(HygonModule *module, const char *path);
int hipModuleGetFunction(HygonFunction *function,
                         HygonModule module,
                         const char *name);
int hipExtModuleLaunchKernel(HygonFunction function,
                             uint32_t global_x,
                             uint32_t global_y,
                             uint32_t global_z,
                             uint32_t local_x,
                             uint32_t local_y,
                             uint32_t local_z,
                             size_t shared_memory,
                             HygonStream stream,
                             void **kernel_params,
                             void **extra,
                             HygonEvent start_event,
                             HygonEvent stop_event,
                             uint32_t flags);
const char *hipGetErrorString(int status);
}
namespace infinicore::op::moe_w4a8_marlin_impl::hygon {
#if defined(ENABLE_ATEN)
void run_vendor(Tensor output,
                const Tensor &input,
                const Tensor &marlin_weight,
                const Tensor &input_scale,
                const Tensor &weight_scale,
                std::optional<Tensor> topk_weights,
                const Tensor &padded_sorted_token_ids,
                const Tensor &expert_ids,
                const Tensor &num_tokens_post_pad,
                int64_t topk);
#endif
namespace {

constexpr uint32_t WAVEFRONT_THREADS = 768;
constexpr uint32_t MARLIN_BLOCK_M = 16;
constexpr const char *KERNEL_NAME = "moe_wi4ai8_marlin_perchannel_Asm_TN_MT128x512x128_WGM1";

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

void check_hygon(int status, const char *operation) {
    if (status != 0) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + hipGetErrorString(status));
    }
}

struct ModuleFunctions {
    HygonModule first_module{};
    HygonModule second_module{};
    HygonFunction first{};
    HygonFunction second{};
};

ModuleFunctions &module_functions() {
    static std::mutex mutex;
    static std::unordered_map<int, ModuleFunctions> functions;
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = functions.find(device);
    if (found != functions.end()) {
        return found->second;
    }

    const char *configured_root = std::getenv("INFINICORE_HYGON_LIGHTOP_MOE_W4A8_ROOT");
    const std::string root = configured_root != nullptr && configured_root[0] != '\0'
                               ? configured_root
                               : "/usr/local/lib/python3.10/dist-packages/lightop/hsa/gfx936/"
                                 "moe_w4a8_channel";
    const std::string prefix = root + "/moe_wi4ai8_marlin_128x512x128_TN_BF16_WGM1_";
    ModuleFunctions loaded;
    check_hygon(hipModuleLoad(&loaded.first_module,
                              (prefix + "FirstStage.co").c_str()),
                "hipModuleLoad FirstStage");
    check_hygon(hipModuleGetFunction(&loaded.first, loaded.first_module,
                                     KERNEL_NAME),
                "hipModuleGetFunction FirstStage");
    check_hygon(hipModuleLoad(&loaded.second_module,
                              (prefix + "SecondStage.co").c_str()),
                "hipModuleLoad SecondStage");
    check_hygon(hipModuleGetFunction(&loaded.second, loaded.second_module,
                                     KERNEL_NAME),
                "hipModuleGetFunction SecondStage");
    return functions.emplace(device, loaded).first->second;
}

__device__ uint32_t weight_permutation(size_t index) {
    constexpr uint32_t interleave[8] = {4, 0, 5, 1, 6, 2, 7, 3};
    const uint32_t reordered = static_cast<uint32_t>((index / 8) * 8) + interleave[index % 8];
    const uint32_t i = reordered / 32;
    const uint32_t remainder = reordered % 32;
    const uint32_t column = remainder / 8;
    const uint32_t row = remainder % 8;
    const uint32_t source_column = (i % 16) * 4 + column;
    const uint32_t source_row = (i / 16) * 8 + row;
    return source_row * 64 + source_column;
}

__global__ void repack_weight_kernel(
    uint32_t *output,
    const uint8_t *input,
    size_t experts,
    size_t n,
    size_t k) {
    const size_t output_columns = n * 4;
    const size_t output_rows = k / 32;
    const size_t total = experts * output_rows * output_columns;
    for (size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output_index < total;
         output_index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t output_column = output_index % output_columns;
        const size_t row_index = output_index / output_columns;
        const size_t k_block = row_index % output_rows;
        const size_t expert = row_index / output_rows;
        uint32_t packed = 0;
        for (size_t nibble_index = 0; nibble_index < 8; ++nibble_index) {
            const size_t linear = output_column * 8 + nibble_index;
            const size_t n_tile = linear / 2048;
            const size_t permuted = linear % 2048;
            const uint32_t source = weight_permutation(permuted);
            const size_t k_index = k_block * 32 + source / 64;
            const size_t n_index = n_tile * 64 + source % 64;
            const size_t packed_k = (k_index / 32) * 16 + (k_index % 16);
            const uint8_t byte = input[(expert * n + n_index) * (k / 2) + packed_k];
            const uint32_t value = (k_index % 32) < 16 ? (byte & 0x0f) : (byte >> 4);
            packed |= value << (4 * nibble_index);
        }
        output[output_index] = packed;
    }
}

constexpr size_t ALIGN_THREADS = 256;

__global__ void align_routes_kernel(
    int32_t *padded_sorted_token_ids,
    int32_t *expert_ids,
    int32_t *num_tokens_post_pad,
    const int32_t *sorted_token_ids,
    const int32_t *tokens_per_expert,
    size_t routes,
    size_t experts,
    int32_t block_size,
    int32_t routing_topk) {
    (void)routing_topk;
    __shared__ int32_t source_prefix[ALIGN_THREADS];
    __shared__ int32_t destination_prefix[ALIGN_THREADS];
    const size_t expert = threadIdx.x;

    const int32_t count = expert < experts ? tokens_per_expert[expert] : 0;
    const int32_t padded_count = ((count + block_size - 1) / block_size) * block_size;
    source_prefix[expert] = count;
    destination_prefix[expert] = padded_count;
    __syncthreads();

    for (size_t shift = 1; shift < ALIGN_THREADS; shift <<= 1) {
        const int32_t source_addend = expert >= shift ? source_prefix[expert - shift] : 0;
        const int32_t destination_addend = expert >= shift ? destination_prefix[expert - shift] : 0;
        __syncthreads();
        source_prefix[expert] += source_addend;
        destination_prefix[expert] += destination_addend;
        __syncthreads();
    }

    if (expert >= experts) {
        return;
    }
    const int32_t source_offset = expert == 0 ? 0 : source_prefix[expert - 1];
    const int32_t destination_offset = expert == 0 ? 0 : destination_prefix[expert - 1];
    for (int32_t index = 0; index < padded_count; ++index) {
        padded_sorted_token_ids[destination_offset + index] = index < count ? sorted_token_ids[source_offset + index]
                                                                            : static_cast<int32_t>(routes);
    }
    for (int32_t index = 0; index < padded_count / block_size; ++index) {
        expert_ids[destination_offset / block_size + index] = static_cast<int32_t>(expert);
    }
    if (expert + 1 == experts) {
        num_tokens_post_pad[0] = destination_offset + padded_count;
    }
}

__global__ void align_routes_fallback_kernel(
    int32_t *padded_sorted_token_ids,
    int32_t *expert_ids,
    int32_t *num_tokens_post_pad,
    const int32_t *sorted_token_ids,
    const int32_t *tokens_per_expert,
    size_t routes,
    size_t experts,
    int32_t block_size) {
    for (size_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
        int32_t source_offset = 0;
        int32_t destination_offset = 0;
        for (size_t previous = 0; previous < expert; ++previous) {
            const int32_t count = tokens_per_expert[previous];
            source_offset += count;
            destination_offset += ((count + block_size - 1) / block_size) * block_size;
        }
        const int32_t count = tokens_per_expert[expert];
        const int32_t padded_count = ((count + block_size - 1) / block_size) * block_size;
        for (int32_t index = 0; index < padded_count; ++index) {
            padded_sorted_token_ids[destination_offset + index] = index < count ? sorted_token_ids[source_offset + index]
                                                                                : static_cast<int32_t>(routes);
        }
        for (int32_t index = 0; index < padded_count / block_size; ++index) {
            expert_ids[destination_offset / block_size + index] = static_cast<int32_t>(expert);
        }
        if (expert + 1 == experts) {
            num_tokens_post_pad[0] = destination_offset + padded_count;
        }
    }
}

struct alignas(8) KernelArguments {
    uint32_t num_wg0;
    uint32_t num_wg1;
    void *output;
    const void *weight;
    const void *input;
    const void *weight_scale;
    const void *input_scale;
    const void *topk_weights;
    const void *padded_sorted_token_ids;
    const void *expert_ids;
    const void *num_tokens_post_pad;
    uint32_t num_experts;
    uint32_t size_m;
    uint32_t size_n;
    uint32_t size_k;
    uint32_t stride_asm;
    uint32_t stride_ask;
    uint32_t stride_bse;
    uint32_t stride_bn;
    uint32_t stride_bk;
    uint32_t topk;
    float topk_reciprocal;
    uint32_t num_full_blocks;
    uint32_t wgm_remainder;
    uint32_t magic_wgm_remainder;
    void *debug_address;
};
static_assert(sizeof(KernelArguments) == 144);
static_assert(offsetof(KernelArguments, num_experts) == 80);
static_assert(offsetof(KernelArguments, debug_address) == 136);

} // namespace

void prepare_weight(Tensor output, const Tensor &input) {
    (void)module_functions();
    const size_t experts = input->size(0);
    const size_t n = input->size(1);
    const size_t k = input->size(2) * 2;
    const size_t output_elements = input->numel() / sizeof(uint32_t);
    constexpr size_t threads = 256;
    const size_t blocks = (output_elements + threads - 1) / threads;
    repack_weight_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                           static_cast<cudaStream_t>(context::getStream())>>>(
        reinterpret_cast<uint32_t *>(output->data()),
        reinterpret_cast<const uint8_t *>(input->data()), experts, n, k);
    check_cuda(cudaGetLastError(), "repack_weight_kernel");
}

void align_routes(Tensor padded_sorted_token_ids,
                  Tensor expert_ids,
                  Tensor num_tokens_post_pad,
                  const Tensor &sorted_token_ids,
                  const Tensor &tokens_per_expert,
                  int64_t block_size,
                  int64_t routing_topk) {
    const auto stream = static_cast<cudaStream_t>(context::getStream());
    if (tokens_per_expert->numel() <= ALIGN_THREADS) {
        align_routes_kernel<<<1, ALIGN_THREADS, 0, stream>>>(
            reinterpret_cast<int32_t *>(padded_sorted_token_ids->data()),
            reinterpret_cast<int32_t *>(expert_ids->data()),
            reinterpret_cast<int32_t *>(num_tokens_post_pad->data()),
            reinterpret_cast<const int32_t *>(sorted_token_ids->data()),
            reinterpret_cast<const int32_t *>(tokens_per_expert->data()),
            sorted_token_ids->numel(), tokens_per_expert->numel(),
            static_cast<int32_t>(block_size),
            static_cast<int32_t>(routing_topk));
    } else {
        align_routes_fallback_kernel<<<1, ALIGN_THREADS, 0, stream>>>(
            reinterpret_cast<int32_t *>(padded_sorted_token_ids->data()),
            reinterpret_cast<int32_t *>(expert_ids->data()),
            reinterpret_cast<int32_t *>(num_tokens_post_pad->data()),
            reinterpret_cast<const int32_t *>(sorted_token_ids->data()),
            reinterpret_cast<const int32_t *>(tokens_per_expert->data()),
            sorted_token_ids->numel(), tokens_per_expert->numel(),
            static_cast<int32_t>(block_size));
    }
    check_cuda(cudaGetLastError(), "align_routes_kernel");
}

void run(Tensor output,
         const Tensor &input,
         const Tensor &marlin_weight,
         const Tensor &input_scale,
         const Tensor &weight_scale,
         std::optional<Tensor> topk_weights,
         const Tensor &padded_sorted_token_ids,
         const Tensor &expert_ids,
         const Tensor &num_tokens_post_pad,
         int64_t topk,
         int64_t routing_topk) {
#if defined(ENABLE_ATEN)
    (void)routing_topk;
    run_vendor(output, input, marlin_weight, input_scale, weight_scale,
               topk_weights, padded_sorted_token_ids, expert_ids,
               num_tokens_post_pad, topk);
#else
    auto &functions = module_functions();
    const uint32_t m = static_cast<uint32_t>(input->size(0));
    const uint32_t n = static_cast<uint32_t>(output->size(1));
    const uint32_t k = static_cast<uint32_t>(input->size(1));
    const uint32_t num_wg0 = (n + 511) / 512;
    const uint32_t grid_z = static_cast<uint32_t>(
        padded_sorted_token_ids->numel() / MARLIN_BLOCK_M);
    KernelArguments arguments{
        num_wg0,
        1,
        output->data(),
        marlin_weight->data(),
        input->data(),
        weight_scale->data(),
        input_scale->data(),
        topk_weights ? (*topk_weights)->data() : nullptr,
        padded_sorted_token_ids->data(),
        expert_ids->data(),
        num_tokens_post_pad->data(),
        static_cast<uint32_t>(marlin_weight->size(0)),
        m,
        n,
        k,
        1,
        1,
        n,
        n * 16,
        k / 32,
        static_cast<uint32_t>(topk),
        1.0f / static_cast<float>(topk),
        0,
        1,
        0x80000001u,
        nullptr,
    };
    size_t argument_size = sizeof(arguments);
    void *extra[] = {
        reinterpret_cast<void *>(1),
        &arguments,
        reinterpret_cast<void *>(2),
        &argument_size,
        reinterpret_cast<void *>(3),
    };
    const HygonFunction function = n <= 512 ? functions.first
                                            : functions.second;
    check_hygon(hipExtModuleLaunchKernel(
                    function,
                    num_wg0 * WAVEFRONT_THREADS, 1, grid_z,
                    WAVEFRONT_THREADS, 1, 1,
                    0, static_cast<HygonStream>(context::getStream()),
                    nullptr, extra, nullptr, nullptr, 0),
                "hipExtModuleLaunchKernel moe_w4a8_marlin");
#endif
}

} // namespace infinicore::op::moe_w4a8_marlin_impl::hygon
#endif
