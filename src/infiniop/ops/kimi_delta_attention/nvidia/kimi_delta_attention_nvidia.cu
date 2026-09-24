#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "kimi_delta_attention_nvidia.cuh"

#include "../cuda/kernel.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef ENABLE_HYGON_API
#include <cstdlib>
#include <dlfcn.h>
#endif

#include <cstdint>
#include <limits>

namespace op::kimi_delta_attention::nvidia {

#ifdef ENABLE_HYGON_API
namespace {

constexpr size_t FLASH_KDA_DIM = 128;
constexpr size_t FLASH_KDA_TILE_SIZE = 16;
constexpr size_t FLASH_KDA_TILE_WORKSPACE_BYTES = 13824;
constexpr size_t FLASH_KDA_WORKSPACE_ALIGNMENT = 256;
constexpr float LOG2_E = 1.4426950408889634f;
constexpr const char *FLASH_KDA_VARLEN_SYMBOL = "_Z10launch_fwdILi128ELb1ELb1ELb0ELb1ELb0EEvPK14__hip_bfloat16S2_"
                                                "S2_S2_S2_PKvfPvPS0_S5_iiiiPKlPKfSA_fiPKiS5_P12ihipStream_t";

using FlashKdaFn = void (*)(const void *q,
                            const void *k,
                            const void *v,
                            const void *g,
                            const void *beta,
                            const void *initial_state,
                            float scale,
                            void *final_state,
                            void *out,
                            void *workspace,
                            int total_tiles,
                            int total_tokens,
                            int num_heads,
                            int num_sequences,
                            const int64_t *cu_seqlens,
                            const float *A_log,
                            const float *dt_bias,
                            float gate_scale,
                            int block_v,
                            const int *state_indices,
                            void *intermediate_states,
                            cudaStream_t stream);

struct FlashKdaWorkspace {
    size_t initial_state_offset;
    size_t final_state_offset;
    size_t cu_seqlens_offset;
    size_t q_offset;
    size_t k_offset;
    size_t v_offset;
    size_t kernel_offset;
    size_t total_size;
};

bool has_dense_flash_kda_qkv(const KimiDeltaAttentionInfo &info);

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

size_t flash_kda_total_tiles(const KimiDeltaAttentionInfo &info) {
    return (info.total_tokens + FLASH_KDA_TILE_SIZE - 1)
             / FLASH_KDA_TILE_SIZE
         + info.B;
}

FlashKdaWorkspace
make_flash_kda_workspace(const KimiDeltaAttentionInfo &info) {
    const size_t compact_state_bytes = info.B * info.H * FLASH_KDA_DIM
                                     * FLASH_KDA_DIM
                                     * sizeof(__nv_bfloat16);
    const size_t kernel_workspace_bytes = info.H
                                        * flash_kda_total_tiles(info)
                                        * FLASH_KDA_TILE_WORKSPACE_BYTES;

    FlashKdaWorkspace layout{};
    size_t offset = 0;
    layout.initial_state_offset = offset;
    offset = align_up(offset + compact_state_bytes,
                      FLASH_KDA_WORKSPACE_ALIGNMENT);
    layout.final_state_offset = offset;
    offset = align_up(offset + compact_state_bytes,
                      FLASH_KDA_WORKSPACE_ALIGNMENT);
    layout.cu_seqlens_offset = offset;
    if (info.cu_seqlens_dtype == INFINI_DTYPE_I32) {
        offset = align_up(offset + (info.B + 1) * sizeof(int64_t),
                          FLASH_KDA_WORKSPACE_ALIGNMENT);
    }
    layout.q_offset = offset;
    layout.k_offset = offset;
    layout.v_offset = offset;
    if (!has_dense_flash_kda_qkv(info)) {
        const size_t tensor_bytes = info.total_tokens * info.H * info.D
                                  * sizeof(__nv_bfloat16);
        layout.q_offset = offset;
        offset = align_up(offset + tensor_bytes,
                          FLASH_KDA_WORKSPACE_ALIGNMENT);
        layout.k_offset = offset;
        offset = align_up(offset + tensor_bytes,
                          FLASH_KDA_WORKSPACE_ALIGNMENT);
        layout.v_offset = offset;
        offset = align_up(offset + tensor_bytes,
                          FLASH_KDA_WORKSPACE_ALIGNMENT);
    }
    layout.kernel_offset = offset;
    layout.total_size = offset + kernel_workspace_bytes;
    return layout;
}

bool is_dense_4d(const std::vector<ptrdiff_t> &strides,
                 size_t dim1,
                 size_t dim2,
                 size_t dim3) {
    return strides.size() == 4 && strides[3] == 1
        && strides[2] == static_cast<ptrdiff_t>(dim3)
        && strides[1] == static_cast<ptrdiff_t>(dim2 * dim3)
        && strides[0] == static_cast<ptrdiff_t>(dim1 * dim2 * dim3);
}

bool is_packable_flash_kda_input(const std::vector<ptrdiff_t> &strides,
                                 size_t dim) {
    return strides.size() == 4 && strides[3] == 1
        && strides[2] == static_cast<ptrdiff_t>(dim)
        && strides[1] >= static_cast<ptrdiff_t>(dim);
}

bool has_dense_flash_kda_qkv(const KimiDeltaAttentionInfo &info) {
    return is_dense_4d(
               info.q_strides, info.total_tokens, info.H, info.D)
        && is_dense_4d(
               info.k_strides, info.total_tokens, info.H, info.D)
        && is_dense_4d(
               info.v_strides, info.total_tokens, info.H, info.D);
}

bool is_flash_kda_eligible(const KimiDeltaAttentionInfo &info) {
    if (info.is_decode || !info.has_cu_seqlens
        || info.data_dtype != INFINI_DTYPE_BF16
        || info.gate_dtype != INFINI_DTYPE_BF16
        || info.D != FLASH_KDA_DIM || !info.use_qk_l2norm
        || info.total_tokens > static_cast<size_t>(std::numeric_limits<int>::max())
        || info.H > static_cast<size_t>(std::numeric_limits<int>::max())
        || info.B > static_cast<size_t>(std::numeric_limits<int>::max())
        || flash_kda_total_tiles(info)
               > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    const bool packable_qkv = is_packable_flash_kda_input(
                                  info.q_strides, info.D)
                           && is_packable_flash_kda_input(
                                  info.k_strides, info.D)
                           && is_packable_flash_kda_input(
                                  info.v_strides, info.D);
    const bool dense_g = is_dense_4d(
        info.g_strides, info.total_tokens, info.H, info.D);
    const bool dense_out = is_dense_4d(
        info.out_strides, info.total_tokens, info.H, info.D);
    const bool dense_state = is_dense_4d(
        info.initial_state_strides, info.H, info.D, info.D);
    const bool dense_final = info.final_state_strides.empty()
                          || is_dense_4d(
                                 info.final_state_strides,
                                 info.H,
                                 info.D,
                                 info.D);
    const bool dense_beta = info.beta_strides.size() == 3
                         && info.beta_strides[2] == 1
                         && info.beta_strides[1]
                                == static_cast<ptrdiff_t>(info.H)
                         && info.beta_strides[0]
                                == static_cast<ptrdiff_t>(
                                    info.total_tokens * info.H);
    return packable_qkv && dense_g && dense_out && dense_state && dense_final
        && dense_beta && info.A_log_strides.size() == 1
        && info.A_log_strides[0] == 1 && info.dt_bias_strides.size() == 2
        && info.dt_bias_strides[1] == 1
        && info.dt_bias_strides[0] == static_cast<ptrdiff_t>(info.D);
}

void *load_flash_kda_library() {
    if (const char *disabled = std::getenv("INFINICORE_DISABLE_FLASH_KDA");
        disabled != nullptr && disabled[0] != '\0' && disabled[0] != '0') {
        return nullptr;
    }

    if (const char *path = std::getenv("INFINICORE_FLASH_KDA_LIBRARY")) {
        if (void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL)) {
            return handle;
        }
    }

    const char *candidates[] = {
        "flash_kda_C.cpython-310-x86_64-linux-gnu.so",
        "/usr/local/lib/python3.10/dist-packages/flash_kda_C.cpython-310-x86_64-linux-gnu.so",
        "flash_kda_C.cpython-312-x86_64-linux-gnu.so",
        "/usr/local/lib/python3.12/dist-packages/flash_kda_C.cpython-312-x86_64-linux-gnu.so",
    };
    for (const char *candidate : candidates) {
        if (void *handle = dlopen(candidate, RTLD_NOW | RTLD_GLOBAL)) {
            return handle;
        }
    }
    return nullptr;
}

FlashKdaFn get_flash_kda_launcher() {
    // The Hygon FlashKDA package exposes a pointer-only launcher in addition
    // to its Torch binding. Keep this dependency optional so other Hygon
    // installations continue to use the portable InfiniOP implementation.
    static FlashKdaFn launcher = []() -> FlashKdaFn {
        if (void *fn = dlsym(RTLD_DEFAULT, FLASH_KDA_VARLEN_SYMBOL)) {
            return reinterpret_cast<FlashKdaFn>(fn);
        }
        static void *handle = load_flash_kda_library();
        if (handle == nullptr) {
            return nullptr;
        }
        return reinterpret_cast<FlashKdaFn>(
            dlsym(handle, FLASH_KDA_VARLEN_SYMBOL));
    }();
    return launcher;
}

__forceinline__ __device__ int64_t
load_optional_index(const void *indices,
                    bool is_i64,
                    size_t index,
                    int64_t fallback) {
    if (indices == nullptr) {
        return fallback;
    }
    return is_i64 ? static_cast<const int64_t *>(indices)[index]
                  : static_cast<const int32_t *>(indices)[index];
}

INFINIOP_CUDA_KERNEL gather_flash_kda_state(
    __nv_bfloat16 *compact_state,
    const __nv_bfloat16 *state_pool,
    const void *indices,
    bool indices_i64,
    size_t num_sequences,
    size_t num_heads,
    size_t pool_size) {
    const size_t state_size = num_heads * FLASH_KDA_DIM * FLASH_KDA_DIM;
    const size_t total = num_sequences * state_size;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += blockDim.x * gridDim.x) {
        const size_t request = i / state_size;
        const size_t state_offset = i % state_size;
        const int64_t pool_index = load_optional_index(
            indices, indices_i64, request, static_cast<int64_t>(request));
        compact_state[i] = pool_index >= 0
                                && static_cast<size_t>(pool_index) < pool_size
                             ? state_pool[pool_index * state_size + state_offset]
                             : __float2bfloat16(0.0f);
    }
}

INFINIOP_CUDA_KERNEL scatter_flash_kda_state(
    __nv_bfloat16 *state_pool,
    const __nv_bfloat16 *compact_state,
    const void *indices,
    bool indices_i64,
    size_t num_sequences,
    size_t num_heads,
    size_t pool_size) {
    const size_t state_size = num_heads * FLASH_KDA_DIM * FLASH_KDA_DIM;
    const size_t total = num_sequences * state_size;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += blockDim.x * gridDim.x) {
        const size_t request = i / state_size;
        const size_t state_offset = i % state_size;
        const int64_t pool_index = load_optional_index(
            indices, indices_i64, request, static_cast<int64_t>(request));
        if (pool_index >= 0 && static_cast<size_t>(pool_index) < pool_size) {
            state_pool[pool_index * state_size + state_offset]
                = compact_state[i];
        }
    }
}

INFINIOP_CUDA_KERNEL convert_cu_seqlens_i32_to_i64(
    int64_t *destination,
    const int32_t *source,
    size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        destination[i] = source[i];
    }
}

INFINIOP_CUDA_KERNEL pack_flash_kda_qkv(
    __nv_bfloat16 *packed_q,
    __nv_bfloat16 *packed_k,
    __nv_bfloat16 *packed_v,
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    size_t total_tokens,
    size_t num_heads,
    ptrdiff_t q_token_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_token_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_token_stride,
    ptrdiff_t v_head_stride) {
    const size_t total = total_tokens * num_heads * FLASH_KDA_DIM;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += blockDim.x * gridDim.x) {
        const size_t dim = i % FLASH_KDA_DIM;
        const size_t head = (i / FLASH_KDA_DIM) % num_heads;
        const size_t token = i / (num_heads * FLASH_KDA_DIM);
        packed_q[i] = q[token * q_token_stride + head * q_head_stride + dim];
        packed_k[i] = k[token * k_token_stride + head * k_head_stride + dim];
        packed_v[i] = v[token * v_token_stride + head * v_head_stride + dim];
    }
}

INFINIOP_CUDA_KERNEL zero_invalid_flash_kda_outputs(
    __nv_bfloat16 *out,
    const int64_t *cu_seqlens,
    const void *initial_indices,
    const void *final_indices,
    bool initial_indices_i64,
    bool final_indices_i64,
    size_t num_sequences,
    size_t num_heads,
    size_t pool_size) {
    const size_t request = blockIdx.x;
    if (request >= num_sequences) {
        return;
    }
    const int64_t initial_index = load_optional_index(
        initial_indices,
        initial_indices_i64,
        request,
        static_cast<int64_t>(request));
    const int64_t final_index = load_optional_index(
        final_indices,
        final_indices_i64,
        request,
        static_cast<int64_t>(request));
    if (initial_index >= 0 && static_cast<size_t>(initial_index) < pool_size
        && final_index >= 0
        && static_cast<size_t>(final_index) < pool_size) {
        return;
    }

    const size_t begin = static_cast<size_t>(cu_seqlens[request]);
    const size_t end = static_cast<size_t>(cu_seqlens[request + 1]);
    const size_t count = (end - begin) * num_heads * FLASH_KDA_DIM;
    const size_t offset = begin * num_heads * FLASH_KDA_DIM;
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        out[offset + i] = __float2bfloat16(0.0f);
    }
}

infiniStatus_t launch_flash_kda(
    FlashKdaFn launcher,
    const KimiDeltaAttentionInfo &info,
    void *workspace,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *A_log,
    const void *dt_bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    cudaStream_t stream) {
    if (workspace == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    const auto layout = make_flash_kda_workspace(info);
    auto *workspace_bytes = static_cast<uint8_t *>(workspace);
    auto *compact_initial = reinterpret_cast<__nv_bfloat16 *>(
        workspace_bytes + layout.initial_state_offset);
    auto *compact_final = reinterpret_cast<__nv_bfloat16 *>(
        workspace_bytes + layout.final_state_offset);
    auto *cu_i64 = info.cu_seqlens_dtype == INFINI_DTYPE_I64
                     ? static_cast<const int64_t *>(cu_seqlens)
                     : reinterpret_cast<int64_t *>(
                         workspace_bytes + layout.cu_seqlens_offset);
    void *kernel_workspace = workspace_bytes + layout.kernel_offset;

    constexpr size_t threads = 256;
    const size_t state_elements = info.B * info.H * FLASH_KDA_DIM
                                * FLASH_KDA_DIM;
    const size_t state_blocks = (state_elements + threads - 1) / threads;
    gather_flash_kda_state<<<state_blocks, threads, 0, stream>>>(
        compact_initial,
        static_cast<const __nv_bfloat16 *>(initial_state),
        initial_state_indices,
        info.initial_state_indices_dtype == INFINI_DTYPE_I64,
        info.B,
        info.H,
        info.pool_size);

    if (info.cu_seqlens_dtype == INFINI_DTYPE_I32) {
        const size_t count = info.B + 1;
        convert_cu_seqlens_i32_to_i64<<<
            (count + threads - 1) / threads,
            threads,
            0,
            stream>>>(
            const_cast<int64_t *>(cu_i64),
            static_cast<const int32_t *>(cu_seqlens),
            count);
    }

    const void *flash_q = q;
    const void *flash_k = k;
    const void *flash_v = v;
    if (!has_dense_flash_kda_qkv(info)) {
        // Kimi's short convolution returns packed QKV views. Repack all three
        // views in one device pass because the vendor kernel requires dense
        // Q/K/V tensors.
        auto *packed_q = reinterpret_cast<__nv_bfloat16 *>(
            workspace_bytes + layout.q_offset);
        auto *packed_k = reinterpret_cast<__nv_bfloat16 *>(
            workspace_bytes + layout.k_offset);
        auto *packed_v = reinterpret_cast<__nv_bfloat16 *>(
            workspace_bytes + layout.v_offset);
        const size_t qkv_elements = info.total_tokens * info.H * info.D;
        pack_flash_kda_qkv<<<
            (qkv_elements + threads - 1) / threads,
            threads,
            0,
            stream>>>(
            packed_q,
            packed_k,
            packed_v,
            static_cast<const __nv_bfloat16 *>(q),
            static_cast<const __nv_bfloat16 *>(k),
            static_cast<const __nv_bfloat16 *>(v),
            info.total_tokens,
            info.H,
            info.q_strides[1],
            info.q_strides[2],
            info.k_strides[1],
            info.k_strides[2],
            info.v_strides[1],
            info.v_strides[2]);
        flash_q = packed_q;
        flash_k = packed_k;
        flash_v = packed_v;
    }

    launcher(
        flash_q,
        flash_k,
        flash_v,
        g,
        beta,
        compact_initial,
        info.scale,
        compact_final,
        out,
        kernel_workspace,
        static_cast<int>(flash_kda_total_tiles(info)),
        static_cast<int>(info.total_tokens),
        static_cast<int>(info.H),
        static_cast<int>(info.B),
        cu_i64,
        static_cast<const float *>(A_log),
        static_cast<const float *>(dt_bias),
        info.lower_bound * LOG2_E,
        32,
        nullptr,
        nullptr,
        stream);

    auto *final_pool = info.has_final_state_indices
                         ? static_cast<__nv_bfloat16 *>(initial_state)
                         : static_cast<__nv_bfloat16 *>(final_state);
    const size_t final_pool_size = info.has_final_state_indices
                                     ? info.pool_size
                                     : info.B;
    scatter_flash_kda_state<<<state_blocks, threads, 0, stream>>>(
        final_pool,
        compact_final,
        final_state_indices,
        info.final_state_indices_dtype == INFINI_DTYPE_I64,
        info.B,
        info.H,
        final_pool_size);

    if (info.has_initial_state_indices || info.has_final_state_indices) {
        zero_invalid_flash_kda_outputs<<<info.B, threads, 0, stream>>>(
            static_cast<__nv_bfloat16 *>(out),
            cu_i64,
            initial_state_indices,
            final_state_indices,
            info.initial_state_indices_dtype == INFINI_DTYPE_I64,
            info.final_state_indices_dtype == INFINI_DTYPE_I64,
            info.B,
            info.H,
            info.pool_size);
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace
#endif

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t A_log_desc,
    infiniopTensorDescriptor_t dt_bias_desc,
    infiniopTensorDescriptor_t cu_seqlens_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc,
    float scale,
    float lower_bound,
    bool use_qk_l2norm) {

    auto info = KimiDeltaAttentionInfo::create(
        out_desc,
        initial_state_desc,
        final_state_desc,
        q_desc,
        k_desc,
        v_desc,
        g_desc,
        beta_desc,
        A_log_desc,
        dt_bias_desc,
        cu_seqlens_desc,
        initial_state_indices_desc,
        final_state_indices_desc,
        scale,
        lower_bound,
        use_qk_l2norm);
    CHECK_RESULT(info);

    auto info_value = info.take();
    size_t workspace_size = 0;
#ifdef ENABLE_HYGON_API
    if (handle->device == INFINI_DEVICE_HYGON
        && is_flash_kda_eligible(info_value)
        && get_flash_kda_launcher() != nullptr) {
        workspace_size = make_flash_kda_workspace(info_value).total_size;
    }
#endif

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info_value,
        workspace_size,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tgate>
static infiniStatus_t launch_warp_sequence(const KimiDeltaAttentionInfo &info,
                                           void *out,
                                           void *initial_state,
                                           void *final_state,
                                           const void *q,
                                           const void *k,
                                           const void *v,
                                           const void *g,
                                           const void *beta,
                                           const void *A_log,
                                           const void *dt_bias,
                                           const void *cu_seqlens,
                                           const void *initial_state_indices,
                                           const void *final_state_indices,
                                           cudaStream_t stream) {
    constexpr size_t D = 128;
    constexpr size_t WARP_SIZE = INFINIOP_RECURRENT_DELTA_RULE_WARP_SIZE;
    constexpr size_t WARPS_PER_BLOCK = 256 / WARP_SIZE;
    constexpr size_t NUM_THREADS = WARPS_PER_BLOCK * WARP_SIZE;
    const dim3 grid(
        static_cast<uint32_t>(info.B),
        static_cast<uint32_t>(info.H),
        static_cast<uint32_t>((D + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    const dim3 block(NUM_THREADS);
    const size_t shared = (D * 3 + NUM_THREADS + 1) * sizeof(float);

    kimiDeltaAttentionWarpCudaKernel<Tdata, Tgate, float, D, WARPS_PER_BLOCK>
        <<<grid, block, shared, stream>>>(
            static_cast<Tdata *>(out),
            static_cast<Tdata *>(initial_state),
            static_cast<Tdata *>(final_state),
            static_cast<const Tdata *>(q),
            static_cast<const Tdata *>(k),
            static_cast<const Tdata *>(v),
            static_cast<const Tgate *>(g),
            static_cast<const Tgate *>(beta),
            static_cast<const float *>(A_log),
            static_cast<const float *>(dt_bias),
            cu_seqlens,
            initial_state_indices,
            final_state_indices,
            info.cu_seqlens_dtype == INFINI_DTYPE_I64,
            info.initial_state_indices_dtype == INFINI_DTYPE_I64,
            info.final_state_indices_dtype == INFINI_DTYPE_I64,
            info.use_qk_l2norm,
            info.has_cu_seqlens,
            info.indexed_state_pool,
            info.total_tokens,
            info.pool_size,
            info.scale,
            info.lower_bound,
            info.out_strides[0],
            info.out_strides[1],
            info.out_strides[2],
            info.initial_state_strides[0],
            info.initial_state_strides[1],
            info.initial_state_strides[2],
            info.initial_state_strides[3],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[0],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[1],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[2],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[3],
            info.q_strides[0],
            info.q_strides[1],
            info.q_strides[2],
            info.k_strides[0],
            info.k_strides[1],
            info.k_strides[2],
            info.v_strides[0],
            info.v_strides[1],
            info.v_strides[2],
            info.g_strides[0],
            info.g_strides[1],
            info.g_strides[2],
            info.beta_strides[0],
            info.beta_strides[1],
            info.beta_strides[2],
            info.A_log_strides[0],
            info.dt_bias_strides[0]);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tgate>
static infiniStatus_t launch_row_streaming_decode(
    const KimiDeltaAttentionInfo &info,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *A_log,
    const void *dt_bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    cudaStream_t stream) {
    constexpr size_t D = 128;
    constexpr size_t WARP_SIZE = INFINIOP_RECURRENT_DELTA_RULE_WARP_SIZE;
    constexpr size_t WARPS_PER_BLOCK = 256 / WARP_SIZE;
    constexpr size_t NUM_THREADS = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr size_t ROW_BLOCKS_PER_HEAD = 4;
    const dim3 grid(static_cast<uint32_t>(
        info.B * info.H * ROW_BLOCKS_PER_HEAD));
    const dim3 block(NUM_THREADS);

    kimiDeltaAttentionRowStreamingDecodeCudaKernel<
        Tdata,
        Tgate,
        D,
        WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
        static_cast<Tdata *>(out),
        static_cast<Tdata *>(initial_state),
        static_cast<Tdata *>(final_state),
        static_cast<const Tdata *>(q),
        static_cast<const Tdata *>(k),
        static_cast<const Tdata *>(v),
        static_cast<const Tgate *>(g),
        static_cast<const Tgate *>(beta),
        static_cast<const float *>(A_log),
        static_cast<const float *>(dt_bias),
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
        info.cu_seqlens_dtype == INFINI_DTYPE_I64,
        info.initial_state_indices_dtype == INFINI_DTYPE_I64,
        info.final_state_indices_dtype == INFINI_DTYPE_I64,
        info.use_qk_l2norm,
        info.has_cu_seqlens,
        info.indexed_state_pool,
        info.H,
        ROW_BLOCKS_PER_HEAD,
        info.pool_size,
        info.scale,
        info.lower_bound,
        info.out_strides[0],
        info.out_strides[1],
        info.out_strides[2],
        info.initial_state_strides[0],
        info.initial_state_strides[1],
        info.initial_state_strides[2],
        info.initial_state_strides[3],
        info.final_state_strides.empty() ? 0 : info.final_state_strides[0],
        info.final_state_strides.empty() ? 0 : info.final_state_strides[1],
        info.final_state_strides.empty() ? 0 : info.final_state_strides[2],
        info.final_state_strides.empty() ? 0 : info.final_state_strides[3],
        info.q_strides[0],
        info.q_strides[1],
        info.q_strides[2],
        info.k_strides[0],
        info.k_strides[1],
        info.k_strides[2],
        info.v_strides[0],
        info.v_strides[1],
        info.v_strides[2],
        info.g_strides[0],
        info.g_strides[1],
        info.g_strides[2],
        info.beta_strides[0],
        info.beta_strides[1],
        info.beta_strides[2],
        info.A_log_strides[0],
        info.dt_bias_strides[0]);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tgate>
static infiniStatus_t launch_fallback(const KimiDeltaAttentionInfo &info,
                                      void *out,
                                      void *initial_state,
                                      void *final_state,
                                      const void *q,
                                      const void *k,
                                      const void *v,
                                      const void *g,
                                      const void *beta,
                                      const void *A_log,
                                      const void *dt_bias,
                                      const void *cu_seqlens,
                                      const void *initial_state_indices,
                                      const void *final_state_indices,
                                      cudaStream_t stream) {
    const bool dense_state_rows = info.initial_state_strides[3] == 1
                               && info.initial_state_strides[2]
                                      == static_cast<ptrdiff_t>(info.D)
                               && (info.final_state_strides.empty()
                                   || (info.final_state_strides[3] == 1
                                       && info.final_state_strides[2]
                                              == static_cast<ptrdiff_t>(info.D)));
    if (info.D == 128 && info.is_decode && dense_state_rows) {
        return launch_row_streaming_decode<Tdata, Tgate>(
            info,
            out,
            initial_state,
            final_state,
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            cu_seqlens,
            initial_state_indices,
            final_state_indices,
            stream);
    }
    if (info.D == 128) {
        return launch_warp_sequence<Tdata, Tgate>(
            info,
            out,
            initial_state,
            final_state,
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            cu_seqlens,
            initial_state_indices,
            final_state_indices,
            stream);
    }

    constexpr int threads = 256;
    dim3 grid(static_cast<uint32_t>(info.B), static_cast<uint32_t>(info.H), static_cast<uint32_t>(info.D));
    size_t shared = info.is_decode ? threads * sizeof(float) : (info.D * 3 + threads) * sizeof(float);

    // Generic fallback for dimensions not covered by the specialized D=128
    // kernels above.
    if (info.is_decode) {
        kimiDeltaAttentionDecodeCudaKernel<Tdata, Tgate><<<grid, threads, shared, stream>>>(
            static_cast<Tdata *>(out),
            static_cast<Tdata *>(initial_state),
            static_cast<Tdata *>(final_state),
            static_cast<const Tdata *>(q),
            static_cast<const Tdata *>(k),
            static_cast<const Tdata *>(v),
            static_cast<const Tgate *>(g),
            static_cast<const Tgate *>(beta),
            static_cast<const float *>(A_log),
            static_cast<const float *>(dt_bias),
            cu_seqlens,
            initial_state_indices,
            final_state_indices,
            info.cu_seqlens_dtype == INFINI_DTYPE_I64,
            info.initial_state_indices_dtype == INFINI_DTYPE_I64,
            info.final_state_indices_dtype == INFINI_DTYPE_I64,
            info.use_qk_l2norm,
            info.has_cu_seqlens,
            info.indexed_state_pool,
            info.D,
            info.pool_size,
            info.scale,
            info.lower_bound,
            info.out_strides[0],
            info.out_strides[1],
            info.out_strides[2],
            info.initial_state_strides[0],
            info.initial_state_strides[1],
            info.initial_state_strides[2],
            info.initial_state_strides[3],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[0],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[1],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[2],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[3],
            info.q_strides[0],
            info.q_strides[1],
            info.q_strides[2],
            info.k_strides[0],
            info.k_strides[1],
            info.k_strides[2],
            info.v_strides[0],
            info.v_strides[1],
            info.v_strides[2],
            info.g_strides[0],
            info.g_strides[1],
            info.g_strides[2],
            info.beta_strides[0],
            info.beta_strides[1],
            info.beta_strides[2],
            info.A_log_strides[0],
            info.dt_bias_strides[0]);
    } else {
        kimiDeltaAttentionRecurrentCudaKernel<Tdata, Tgate><<<grid, threads, shared, stream>>>(
            static_cast<Tdata *>(out),
            static_cast<Tdata *>(initial_state),
            static_cast<Tdata *>(final_state),
            static_cast<const Tdata *>(q),
            static_cast<const Tdata *>(k),
            static_cast<const Tdata *>(v),
            static_cast<const Tgate *>(g),
            static_cast<const Tgate *>(beta),
            static_cast<const float *>(A_log),
            static_cast<const float *>(dt_bias),
            cu_seqlens,
            initial_state_indices,
            final_state_indices,
            info.cu_seqlens_dtype == INFINI_DTYPE_I64,
            info.initial_state_indices_dtype == INFINI_DTYPE_I64,
            info.final_state_indices_dtype == INFINI_DTYPE_I64,
            info.use_qk_l2norm,
            info.has_cu_seqlens,
            info.indexed_state_pool,
            info.T,
            info.D,
            info.pool_size,
            info.scale,
            info.lower_bound,
            info.out_strides[0],
            info.out_strides[1],
            info.out_strides[2],
            info.initial_state_strides[0],
            info.initial_state_strides[1],
            info.initial_state_strides[2],
            info.initial_state_strides[3],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[0],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[1],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[2],
            info.final_state_strides.empty() ? 0 : info.final_state_strides[3],
            info.q_strides[0],
            info.q_strides[1],
            info.q_strides[2],
            info.k_strides[0],
            info.k_strides[1],
            info.k_strides[2],
            info.v_strides[0],
            info.v_strides[1],
            info.v_strides[2],
            info.g_strides[0],
            info.g_strides[1],
            info.g_strides[2],
            info.beta_strides[0],
            info.beta_strides[1],
            info.beta_strides[2],
            info.A_log_strides[0],
            info.dt_bias_strides[0]);
    }
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
static infiniStatus_t launch_for_gate(const KimiDeltaAttentionInfo &info,
                                      void *out,
                                      void *initial_state,
                                      void *final_state,
                                      const void *q,
                                      const void *k,
                                      const void *v,
                                      const void *g,
                                      const void *beta,
                                      const void *A_log,
                                      const void *dt_bias,
                                      const void *cu_seqlens,
                                      const void *initial_state_indices,
                                      const void *final_state_indices,
                                      cudaStream_t stream) {
    switch (info.gate_dtype) {
    case INFINI_DTYPE_F16:
        return launch_fallback<Tdata, half>(info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_BF16:
        return launch_fallback<Tdata, __nv_bfloat16>(info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_F32:
        return launch_fallback<Tdata, float>(info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *A_log,
    const void *dt_bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream_) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.has_cu_seqlens && cu_seqlens == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (_info.has_initial_state_indices && initial_state_indices == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (_info.has_final_state_indices && final_state_indices == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (!_info.has_final_state_indices && final_state == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
#ifdef ENABLE_HYGON_API
    if (_workspace_size != 0 && device_type == INFINI_DEVICE_HYGON
        && is_flash_kda_eligible(_info)) {
        if (auto launcher = get_flash_kda_launcher()) {
            return launch_flash_kda(
                launcher,
                _info,
                workspace,
                out,
                initial_state,
                final_state,
                q,
                k,
                v,
                g,
                beta,
                A_log,
                dt_bias,
                cu_seqlens,
                initial_state_indices,
                final_state_indices,
                stream);
        }
    }
#endif
    switch (_info.data_dtype) {
    case INFINI_DTYPE_F16:
        return launch_for_gate<half>(_info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_BF16:
        return launch_for_gate<__nv_bfloat16>(_info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_F32:
        return launch_for_gate<float>(_info, out, initial_state, final_state, q, k, v, g, beta, A_log, dt_bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::kimi_delta_attention::nvidia
