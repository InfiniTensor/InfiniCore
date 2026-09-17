#pragma once

#include "../info.h"

namespace op::mamba2_scan::cuda {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;

__device__ inline float softplus(float value) {
    return value > 20.0f ? value : log1pf(expf(value));
}

template <typename T>
static __global__ void coefficients(float *coeff, const T *dt, const float *a,
                                    const float *dt_bias, size_t tokens, size_t heads) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < tokens * heads) {
        const float step = softplus(static_cast<float>(dt[i]) + dt_bias[i % heads]);
        coeff[2 * i] = step;
        coeff[2 * i + 1] = expf(step * a[i % heads]);
    }
}

// Each logical 32-lane group owns one channel, including on 64-lane devices.
// Summaries encode an affine state transform for each independent chunk.
template <typename T, bool Summary, bool Chunked>
static __global__ void scan(T *out, const T *x, const T *dt, const T *b, const T *c,
                            const float *a, const float *d, const float *dt_bias,
                            float *state, const int32_t *offsets, const int32_t *initial,
                            const int32_t *final, const float *coeff, float *chunks,
                            float *decays, Mamba2ScanInfo info) {
    const size_t request = blockIdx.y;
    const size_t p_blocks = (info.head_dim + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const size_t h = blockIdx.x / p_blocks;
    const size_t p = (blockIdx.x % p_blocks) * kWarpsPerBlock + threadIdx.x / kWarpSize;
    const size_t lane = threadIdx.x % kWarpSize;
    if (p >= info.head_dim) {
        return;
    }
    const int32_t start = offsets[request], end = offsets[request + 1];
    const int32_t src = initial[request], dst = final[request];
    if (start < 0 || end <= start || static_cast<size_t>(end) > info.tokens
        || src < 0 || static_cast<size_t>(src) >= info.pool_size
        || dst <= 0 || static_cast<size_t>(dst) >= info.pool_size) {
        return;
    }
    const size_t first = start + (Chunked ? blockIdx.z * info.chunk_size : 0);
    if (first >= static_cast<size_t>(end)) {
        return;
    }
    const size_t last = Chunked ? min(first + info.chunk_size, static_cast<size_t>(end)) : end;
    const size_t slot = start / info.chunk_size + request + blockIdx.z;
    const size_t local_state = (h * info.head_dim + p) * info.state_size;
    const size_t chunk_base = slot * (info.heads * info.head_dim * info.state_size) + local_state;
    const size_t source_base = static_cast<size_t>(src) * (info.heads * info.head_dim * info.state_size) + local_state;
    float values[8];
    for (size_t j = 0; j < 8; ++j) {
        const size_t n = lane + j * kWarpSize;
        values[j] = n < info.state_size && !Summary
                      ? (Chunked ? chunks[chunk_base + n] : state[source_base + n])
                      : 0.0f;
    }
    float decay = 1.0f;
    const size_t group = h / (info.heads / info.groups);
    for (size_t t = first; t < last; ++t) {
        const size_t time_head = t * info.heads + h;
        const float step = Chunked ? coeff[2 * time_head]
                                   : softplus(static_cast<float>(dt[time_head]) + dt_bias[h]);
        const float alpha = Chunked ? coeff[2 * time_head + 1] : expf(step * a[h]);
        const size_t x_index = time_head * info.head_dim + p;
        const float input = static_cast<float>(x[x_index]);
        const size_t bc_base = (t * info.groups + group) * info.state_size;
        float y = 0.0f;
        for (size_t j = 0; j < 8; ++j) {
            const size_t n = lane + j * kWarpSize;
            if (n < info.state_size) {
                values[j] = fmaf(alpha, values[j], step * input * static_cast<float>(b[bc_base + n]));
                if constexpr (!Summary) {
                    y = fmaf(values[j], static_cast<float>(c[bc_base + n]), y);
                }
            }
        }
        if constexpr (Summary) {
            decay *= alpha;
        } else {
            for (int delta = 16; delta > 0; delta /= 2) {
                y += __shfl_down_sync(__activemask(), y, delta, kWarpSize);
            }
            if (lane == 0) {
                out[x_index] = static_cast<T>(y + d[h] * input);
            }
        }
    }
    if constexpr (Summary || !Chunked) {
        const size_t target_base = Summary ? chunk_base : static_cast<size_t>(dst) * (info.heads * info.head_dim * info.state_size) + local_state;
        for (size_t j = 0; j < 8; ++j) {
            const size_t n = lane + j * kWarpSize;
            if (n < info.state_size) {
                (Summary ? chunks : state)[target_base + n] = values[j];
            }
        }
        if constexpr (Summary) {
            if (p == 0 && lane == 0) {
                decays[slot * info.heads + h] = decay;
            }
        }
    }
}

// Convert chunk summaries to initial states, then publish the request's final state.
static __global__ void carry(float *chunks, const float *decays, float *state,
                             const int32_t *offsets, const int32_t *initial,
                             const int32_t *final, Mamba2ScanInfo info) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t request = blockIdx.y;
    if (element >= (info.heads * info.head_dim * info.state_size)) {
        return;
    }
    const int32_t start = offsets[request], end = offsets[request + 1];
    const int32_t src = initial[request], dst = final[request];
    if (start < 0 || end <= start || static_cast<size_t>(end) > info.tokens
        || src < 0 || static_cast<size_t>(src) >= info.pool_size
        || dst <= 0 || static_cast<size_t>(dst) >= info.pool_size) {
        return;
    }
    const size_t h = element / (info.head_dim * info.state_size);
    const size_t count = (end - start + info.chunk_size - 1) / info.chunk_size;
    const size_t first_slot = start / info.chunk_size + request;
    float value = state[static_cast<size_t>(src) * (info.heads * info.head_dim * info.state_size) + element];
    for (size_t chunk = 0; chunk < count; ++chunk) {
        const size_t slot = first_slot + chunk;
        const size_t i = slot * (info.heads * info.head_dim * info.state_size) + element;
        const float summary = chunks[i];
        chunks[i] = value;
        value = fmaf(decays[slot * info.heads + h], value, summary);
    }
    state[static_cast<size_t>(dst) * (info.heads * info.head_dim * info.state_size) + element] = value;
}

} // namespace op::mamba2_scan::cuda
