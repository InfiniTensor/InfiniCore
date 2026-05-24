#include "flash_attention_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

// Flash Attention Kernel with Online Softmax
// Each block handles one (batch, head) pair, threads process query positions
// Uses single-pass online softmax algorithm for numerical stability

template <typename T, typename Tcompute, int BLOCK_SIZE>
__global__ void flashAttentionKernel(
    T *out,
    const T *q, const T *k, const T *v,
    const int64_t *total_kv_len,
    float scale,
    int is_causal,
    size_t batch_size, size_t num_q_heads, size_t num_kv_heads,
    size_t seq_len, size_t max_kv_len, size_t head_dim, size_t ngroup,
    // strides
    ptrdiff_t q_stride_b, ptrdiff_t q_stride_h, ptrdiff_t q_stride_s,
    ptrdiff_t k_stride_b, ptrdiff_t k_stride_h, ptrdiff_t k_stride_s,
    ptrdiff_t v_stride_b, ptrdiff_t v_stride_h, ptrdiff_t v_stride_s,
    ptrdiff_t o_stride_b, ptrdiff_t o_stride_h, ptrdiff_t o_stride_s) {

    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // GQA: compute corresponding kv_head
    const int kv_head_idx = head_idx / ngroup;

    // Get actual kv length for this batch
    const int actual_kv_len = static_cast<int>(total_kv_len[batch_idx]);

    // Base pointers
    const T *q_base = q + batch_idx * q_stride_b + head_idx * q_stride_h;
    const T *k_base = k + batch_idx * k_stride_b + kv_head_idx * k_stride_h;
    const T *v_base = v + batch_idx * v_stride_b + kv_head_idx * v_stride_h;
    T *o_base = out + batch_idx * o_stride_b + head_idx * o_stride_h;

    // Each thread processes one or more query positions
    for (int q_pos = tid; q_pos < seq_len; q_pos += BLOCK_SIZE) {
        // Online softmax variables (single pass algorithm)
        Tcompute max_val = -INFINITY;
        Tcompute sum_exp = 0;
        
        // Accumulator for weighted sum of V (online update)
        // Use local array for accumulation to avoid recomputation
        constexpr int MAX_HEAD_DIM = 128;
        Tcompute acc[MAX_HEAD_DIM];
        for (int d = 0; d < head_dim && d < MAX_HEAD_DIM; ++d) {
            acc[d] = 0;
        }

        // Single pass: compute Q*K^T, online softmax, and weighted V sum
        // For causal attention: q_pos here is the local index within the current
        // query window. The absolute position of this query token is:
        //   abs_q_pos = actual_kv_len - seq_len + q_pos
        // so causal mask should be: kv_pos > abs_q_pos
        // (decode: seq_len==1, abs_q_pos==actual_kv_len-1, never mask)
        // (prefill: seq_len>1, abs_q_pos increases from actual_kv_len-seq_len)
        const int abs_q_pos = (int)actual_kv_len - (int)seq_len + q_pos;
        for (int kv_pos = 0; kv_pos < actual_kv_len; ++kv_pos) {
            // Causal mask: mask future tokens relative to absolute q position
            if (is_causal && kv_pos > abs_q_pos) {
                continue;
            }

            // Compute Q * K^T
            Tcompute qk = 0;
            for (int d = 0; d < head_dim; ++d) {
                T q_val = q_base[q_pos * q_stride_s + d];
                T k_val = k_base[kv_pos * k_stride_s + d];
                qk += static_cast<Tcompute>(q_val) * static_cast<Tcompute>(k_val);
            }
            qk *= scale;

            // Online softmax update (FlashAttention v2 style)
            Tcompute new_max = max(max_val, qk);
            
            if (max_val == -INFINITY) {
                // First valid element
                sum_exp = exp_(qk - new_max);
                for (int d = 0; d < head_dim && d < MAX_HEAD_DIM; ++d) {
                    T v_val = v_base[kv_pos * v_stride_s + d];
                    acc[d] = exp_(qk - new_max) * static_cast<Tcompute>(v_val);
                }
            } else {
                // Update existing accumulators
                Tcompute scale_factor = exp_(max_val - new_max);
                sum_exp = sum_exp * scale_factor + exp_(qk - new_max);
                for (int d = 0; d < head_dim && d < MAX_HEAD_DIM; ++d) {
                    T v_val = v_base[kv_pos * v_stride_s + d];
                    acc[d] = acc[d] * scale_factor + exp_(qk - new_max) * static_cast<Tcompute>(v_val);
                }
            }
            
            max_val = new_max;
        }

        // Handle case where all values are masked
        if (max_val == -INFINITY || sum_exp == 0) {
            // Set output to zero (or could copy Q as residual)
            for (int d = 0; d < head_dim; ++d) {
                o_base[q_pos * o_stride_s + d] = T(0);
            }
            continue;
        }

        // Normalize and write output
        Tcompute inv_sum = 1.0 / sum_exp;
        for (int d = 0; d < head_dim; ++d) {
            o_base[q_pos * o_stride_s + d] = static_cast<T>(acc[d] * inv_sum);
        }
    }
}

namespace op::flash_attention::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

Descriptor::Descriptor(
    Opaque *opaque,
    std::vector<size_t> q_shape, std::vector<ptrdiff_t> q_strides,
    std::vector<size_t> k_shape, std::vector<ptrdiff_t> k_strides,
    std::vector<size_t> v_shape, std::vector<ptrdiff_t> v_strides,
    std::vector<size_t> out_shape, std::vector<ptrdiff_t> out_strides,
    infiniDtype_t dtype, float scale, char is_causal,
    size_t batch_size, size_t num_q_heads, size_t num_kv_heads,
    size_t seq_len, size_t max_kv_len, size_t head_dim, size_t ngroup)
    : InfiniopDescriptor{INFINI_DEVICE_NVIDIA, 0},
      _opaque(opaque),
      _q_shape(std::move(q_shape)),
      _k_shape(std::move(k_shape)),
      _v_shape(std::move(v_shape)),
      _out_shape(std::move(out_shape)),
      _q_strides(std::move(q_strides)),
      _k_strides(std::move(k_strides)),
      _v_strides(std::move(v_strides)),
      _out_strides(std::move(out_strides)),
      _dtype(dtype),
      _scale(scale),
      _is_causal(is_causal),
      _batch_size(batch_size),
      _num_q_heads(num_q_heads),
      _num_kv_heads(num_kv_heads),
      _seq_len(seq_len),
      _max_kv_len(max_kv_len),
      _head_dim(head_dim),
      _ngroup(ngroup) {
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t total_kv_len_desc,
    float scale,
    char is_causal) {

    auto handle_nvidia = reinterpret_cast<device::nvidia::Handle *>(handle);

    // Get shapes and strides
    auto q_shape = q_desc->shape();
    auto k_shape = k_desc->shape();
    auto v_shape = v_desc->shape();
    auto out_shape = out_desc->shape();

    auto q_strides = q_desc->strides();
    auto k_strides = k_desc->strides();
    auto v_strides = v_desc->strides();
    auto out_strides = out_desc->strides();

    // Validate shapes: expect [batch, num_heads, seq_len, head_dim]
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4 || out_shape.size() != 4) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t batch_size = q_shape[0];
    size_t num_q_heads = q_shape[1];
    size_t seq_len = q_shape[2];
    size_t head_dim = q_shape[3];

    size_t num_kv_heads = k_shape[1];
    size_t max_kv_len = k_shape[2];

    // Validate consistency
    if (k_shape[0] != batch_size || v_shape[0] != batch_size || out_shape[0] != batch_size) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (k_shape[1] != num_kv_heads || v_shape[1] != num_kv_heads) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (k_shape[2] != max_kv_len || v_shape[2] != max_kv_len) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (k_shape[3] != head_dim || v_shape[3] != head_dim || out_shape[3] != head_dim) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (out_shape[1] != num_q_heads || out_shape[2] != seq_len) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Check GQA grouping
    if (num_q_heads % num_kv_heads != 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    size_t ngroup = num_q_heads / num_kv_heads;

    // Check dtype
    auto dtype = q_desc->dtype();
    if (dtype != k_desc->dtype() || dtype != v_desc->dtype() || dtype != out_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    *desc_ptr = new Descriptor(
        new Opaque{handle_nvidia->internal()},
        q_shape, q_strides,
        k_shape, k_strides,
        v_shape, v_strides,
        out_shape, out_strides,
        dtype, scale, is_causal,
        batch_size, num_q_heads, num_kv_heads,
        seq_len, max_kv_len, head_dim, ngroup);

    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::get_workspace_size() const {
    return 0;
}

template <typename T, typename Tcompute, int BLOCK_SIZE>
infiniStatus_t launchFlashAttentionKernel(
    const T *q, const T *k, const T *v, T *out,
    const int64_t *total_kv_len,
    float scale, int is_causal,
    size_t batch_size, size_t num_q_heads, size_t num_kv_heads,
    size_t seq_len, size_t max_kv_len, size_t head_dim, size_t ngroup,
    ptrdiff_t q_stride_b, ptrdiff_t q_stride_h, ptrdiff_t q_stride_s,
    ptrdiff_t k_stride_b, ptrdiff_t k_stride_h, ptrdiff_t k_stride_s,
    ptrdiff_t v_stride_b, ptrdiff_t v_stride_h, ptrdiff_t v_stride_s,
    ptrdiff_t o_stride_b, ptrdiff_t o_stride_h, ptrdiff_t o_stride_s,
    cudaStream_t stream) {

    dim3 grid(static_cast<uint32_t>(num_q_heads), static_cast<uint32_t>(batch_size), 1);

    flashAttentionKernel<T, Tcompute, BLOCK_SIZE>
        <<<grid, BLOCK_SIZE, 0, stream>>>(
            out, q, k, v, total_kv_len,
            scale, is_causal,
            batch_size, num_q_heads, num_kv_heads,
            seq_len, max_kv_len, head_dim, ngroup,
            q_stride_b, q_stride_h, q_stride_s,
            k_stride_b, k_stride_h, k_stride_s,
            v_stride_b, v_stride_h, v_stride_s,
            o_stride_b, o_stride_h, o_stride_s);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    void *stream_) const {

    cudaStream_t stream = static_cast<cudaStream_t>(stream_);

    // Extract strides
    ptrdiff_t q_stride_b = _q_strides[0];
    ptrdiff_t q_stride_h = _q_strides[1];
    ptrdiff_t q_stride_s = _q_strides[2];

    ptrdiff_t k_stride_b = _k_strides[0];
    ptrdiff_t k_stride_h = _k_strides[1];
    ptrdiff_t k_stride_s = _k_strides[2];

    ptrdiff_t v_stride_b = _v_strides[0];
    ptrdiff_t v_stride_h = _v_strides[1];
    ptrdiff_t v_stride_s = _v_strides[2];

    ptrdiff_t o_stride_b = _out_strides[0];
    ptrdiff_t o_stride_h = _out_strides[1];
    ptrdiff_t o_stride_s = _out_strides[2];

    int block_size = _opaque->internal->maxThreadsPerBlock();
    if (block_size > 256) {
        block_size = 256; // Use smaller block size for better occupancy
    }

    if (_dtype == INFINI_DTYPE_F16) {
        if (block_size >= 256) {
            return launchFlashAttentionKernel<half, float, 256>(
                static_cast<const half *>(q),
                static_cast<const half *>(k),
                static_cast<const half *>(v),
                static_cast<half *>(out),
                static_cast<const int64_t *>(total_kv_len),
                _scale, _is_causal,
                _batch_size, _num_q_heads, _num_kv_heads,
                _seq_len, _max_kv_len, _head_dim, _ngroup,
                q_stride_b, q_stride_h, q_stride_s,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                o_stride_b, o_stride_h, o_stride_s,
                stream);
        } else if (block_size >= 128) {
            return launchFlashAttentionKernel<half, float, 128>(
                static_cast<const half *>(q),
                static_cast<const half *>(k),
                static_cast<const half *>(v),
                static_cast<half *>(out),
                static_cast<const int64_t *>(total_kv_len),
                _scale, _is_causal,
                _batch_size, _num_q_heads, _num_kv_heads,
                _seq_len, _max_kv_len, _head_dim, _ngroup,
                q_stride_b, q_stride_h, q_stride_s,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                o_stride_b, o_stride_h, o_stride_s,
                stream);
        } else {
            return launchFlashAttentionKernel<half, float, 64>(
                static_cast<const half *>(q),
                static_cast<const half *>(k),
                static_cast<const half *>(v),
                static_cast<half *>(out),
                static_cast<const int64_t *>(total_kv_len),
                _scale, _is_causal,
                _batch_size, _num_q_heads, _num_kv_heads,
                _seq_len, _max_kv_len, _head_dim, _ngroup,
                q_stride_b, q_stride_h, q_stride_s,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                o_stride_b, o_stride_h, o_stride_s,
                stream);
        }
    } else if (_dtype == INFINI_DTYPE_BF16) {
        if (block_size >= 256) {
            return launchFlashAttentionKernel<__nv_bfloat16, float, 256>(
                static_cast<const __nv_bfloat16 *>(q),
                static_cast<const __nv_bfloat16 *>(k),
                static_cast<const __nv_bfloat16 *>(v),
                static_cast<__nv_bfloat16 *>(out),
                static_cast<const int64_t *>(total_kv_len),
                _scale, _is_causal,
                _batch_size, _num_q_heads, _num_kv_heads,
                _seq_len, _max_kv_len, _head_dim, _ngroup,
                q_stride_b, q_stride_h, q_stride_s,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                o_stride_b, o_stride_h, o_stride_s,
                stream);
        } else if (block_size >= 128) {
            return launchFlashAttentionKernel<__nv_bfloat16, float, 128>(
                static_cast<const __nv_bfloat16 *>(q),
                static_cast<const __nv_bfloat16 *>(k),
                static_cast<const __nv_bfloat16 *>(v),
                static_cast<__nv_bfloat16 *>(out),
                static_cast<const int64_t *>(total_kv_len),
                _scale, _is_causal,
                _batch_size, _num_q_heads, _num_kv_heads,
                _seq_len, _max_kv_len, _head_dim, _ngroup,
                q_stride_b, q_stride_h, q_stride_s,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                o_stride_b, o_stride_h, o_stride_s,
                stream);
        } else {
            return launchFlashAttentionKernel<__nv_bfloat16, float, 64>(
                static_cast<const __nv_bfloat16 *>(q),
                static_cast<const __nv_bfloat16 *>(k),
                static_cast<const __nv_bfloat16 *>(v),
                static_cast<__nv_bfloat16 *>(out),
                static_cast<const int64_t *>(total_kv_len),
                _scale, _is_causal,
                _batch_size, _num_q_heads, _num_kv_heads,
                _seq_len, _max_kv_len, _head_dim, _ngroup,
                q_stride_b, q_stride_h, q_stride_s,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                o_stride_b, o_stride_h, o_stride_s,
                stream);
        }
    } else if (_dtype == INFINI_DTYPE_F32) {
        if (block_size >= 256) {
            return launchFlashAttentionKernel<float, float, 256>(
                static_cast<const float *>(q),
                static_cast<const float *>(k),
                static_cast<const float *>(v),
                static_cast<float *>(out),
                static_cast<const int64_t *>(total_kv_len),
                _scale, _is_causal,
                _batch_size, _num_q_heads, _num_kv_heads,
                _seq_len, _max_kv_len, _head_dim, _ngroup,
                q_stride_b, q_stride_h, q_stride_s,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                o_stride_b, o_stride_h, o_stride_s,
                stream);
        } else if (block_size >= 128) {
            return launchFlashAttentionKernel<float, float, 128>(
                static_cast<const float *>(q),
                static_cast<const float *>(k),
                static_cast<const float *>(v),
                static_cast<float *>(out),
                static_cast<const int64_t *>(total_kv_len),
                _scale, _is_causal,
                _batch_size, _num_q_heads, _num_kv_heads,
                _seq_len, _max_kv_len, _head_dim, _ngroup,
                q_stride_b, q_stride_h, q_stride_s,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                o_stride_b, o_stride_h, o_stride_s,
                stream);
        } else {
            return launchFlashAttentionKernel<float, float, 64>(
                static_cast<const float *>(q),
                static_cast<const float *>(k),
                static_cast<const float *>(v),
                static_cast<float *>(out),
                static_cast<const int64_t *>(total_kv_len),
                _scale, _is_causal,
                _batch_size, _num_q_heads, _num_kv_heads,
                _seq_len, _max_kv_len, _head_dim, _ngroup,
                q_stride_b, q_stride_h, q_stride_s,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                o_stride_b, o_stride_h, o_stride_s,
                stream);
        }
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::flash_attention::nvidia
