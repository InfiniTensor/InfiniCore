#pragma once

#include "kernel.cuh"

namespace op::mamba2_scan::cuda {

template <typename T, typename Stream>
void launch(const Mamba2ScanInfo &info, void *workspace, void *out,
            const void *x, const void *dt, const void *b, const void *c,
            const void *a, const void *d, const void *dt_bias, void *state,
            const void *offsets, const void *initial, const void *final,
            Stream stream) {
    const dim3 blocks(info.heads * ((info.head_dim + kWarpsPerBlock - 1) / kWarpsPerBlock), info.requests);
    constexpr int threads = kWarpSize * kWarpsPerBlock;
    auto *out_ptr = static_cast<T *>(out);
    const auto *x_ptr = static_cast<const T *>(x), *dt_ptr = static_cast<const T *>(dt);
    const auto *b_ptr = static_cast<const T *>(b), *c_ptr = static_cast<const T *>(c);
    const auto *a_ptr = static_cast<const float *>(a), *d_ptr = static_cast<const float *>(d);
    const auto *bias_ptr = static_cast<const float *>(dt_bias);
    auto *state_ptr = static_cast<float *>(state);
    const auto *offset_ptr = static_cast<const int32_t *>(offsets);
    const auto *init_ptr = static_cast<const int32_t *>(initial), *final_ptr = static_cast<const int32_t *>(final);
    if (info.single_chunk()) {
        scan<T, false, false><<<blocks, threads, 0, stream>>>(out_ptr, x_ptr, dt_ptr, b_ptr, c_ptr, a_ptr, d_ptr, bias_ptr, state_ptr, offset_ptr, init_ptr, final_ptr, nullptr, nullptr, nullptr, info);
    } else {
        auto *coeff = static_cast<float *>(workspace);
        auto *chunks = coeff + 2 * info.tokens * info.heads;
        auto *decays = chunks + info.chunk_slots() * info.state_elements();
        coefficients<T><<<(info.tokens * info.heads + 255) / 256, 256, 0, stream>>>(coeff, dt_ptr, a_ptr, bias_ptr, info.tokens, info.heads);
        const dim3 chunk_blocks(blocks.x, blocks.y, info.max_chunks());
        scan<T, true, true><<<chunk_blocks, threads, 0, stream>>>(out_ptr, x_ptr, dt_ptr, b_ptr, c_ptr, a_ptr, d_ptr, bias_ptr, state_ptr, offset_ptr, init_ptr, final_ptr, coeff, chunks, decays, info);
        carry<<<dim3((info.state_elements() + 255) / 256, info.requests), 256, 0, stream>>>(chunks, decays, state_ptr, offset_ptr, init_ptr, final_ptr, info);
        scan<T, false, true><<<chunk_blocks, threads, 0, stream>>>(out_ptr, x_ptr, dt_ptr, b_ptr, c_ptr, a_ptr, d_ptr, bias_ptr, state_ptr, offset_ptr, init_ptr, final_ptr, coeff, chunks, decays, info);
    }
}
} // namespace op::mamba2_scan::cuda
