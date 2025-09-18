#ifndef __FLASH_ATTENTION_BACKWARD_KERNEL_CUH__
#define __FLASH_ATTENTION_BACKWARD_KERNEL_CUH__

template <typename Tdata>
__device__ void flashAttentionBackwardBlock(
    Tdata *grad_q_, Tdata *grad_k_, Tdata *grad_v_,
    const Tdata *q_, const Tdata *k_, const Tdata *v_,
    const Tdata *out_, const Tdata *grad_out_, const Tdata *l_,
    const float *mask_,
    const size_t seq_len_q, const size_t seq_len_kv,
    const size_t head_dim, const size_t group,
    const size_t B_r, const size_t B_c, const size_t T_r, const size_t T_c,
    const Tdata softmax_scale,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n,
    ptrdiff_t l_stride_b, ptrdiff_t l_stride_s, ptrdiff_t l_stride_n) {

    size_t bx = blockIdx.x;  // batch                          -> batch_size
    size_t by = blockIdx.y;  // q's head index                 -> num_heads
    size_t tx = threadIdx.x; // k's row index within one block -> B_r/B_c

    size_t qo_offset = bx * qo_stride_b + by * qo_stride_n;
    size_t kv_offset = bx * kv_stride_b + by / group * kv_stride_n;
    size_t l_offset = bx * l_stride_b + by * seq_len_q;

    extern __shared__ __align__(sizeof(Tdata)) char shared_mem[];
    Tdata *q_i = reinterpret_cast<Tdata *>(shared_mem);
    Tdata *out_i = reinterpret_cast<Tdata *>(q_i + B_r * head_dim);
    Tdata *grad_out_i = reinterpret_cast<Tdata *>(out_i + B_r * head_dim);

    Tdata *k_j = reinterpret_cast<Tdata *>(grad_out_i + B_r * head_dim);
    Tdata *v_j = reinterpret_cast<Tdata *>(k_j + B_c * head_dim);
    Tdata *grad_k_j = reinterpret_cast<Tdata *>(v_j + B_c * head_dim);
    Tdata *grad_v_j = reinterpret_cast<Tdata *>(grad_k_j + B_c * head_dim);

    Tdata *s_i = reinterpret_cast<Tdata *>(grad_v_j + B_c * head_dim);
    Tdata *grad_s_i = reinterpret_cast<Tdata *>(s_i + B_r * B_c);

    for (size_t j = 0; j < T_c; ++j) {
        // load k_j, v_j and initialize grad_k_j, grad_q_j to 0
        for (size_t x = 0; x < head_dim; ++x) {
            k_j[tx * head_dim + x] = k_[kv_offset + (j * B_c + tx) * kv_stride_s + x];
            v_j[tx * head_dim + x] = v_[kv_offset + (j * B_c + tx) * kv_stride_s + x];
            grad_k_j[tx * head_dim + x] = 0;
            grad_v_j[tx * head_dim + x] = 0;
        }

        for (size_t i = 0; i < T_r; ++i) {
            __syncthreads();
            // load q_i, out_i, grad_out_i
            Tdata D_i = 0;
            for (size_t x = 0; x < head_dim; ++x) {
                q_i[tx * head_dim + x] = q_[qo_offset + (i * B_r + tx) * qo_stride_s + x];
                out_i[tx * head_dim + x] = out_[qo_offset + (i * B_r + tx) * qo_stride_s + x];
                grad_out_i[tx * head_dim + x] = grad_out_[qo_offset + (i * B_r + tx) * qo_stride_s + x];
                D_i += grad_out_i[tx * head_dim + x] * out_i[tx * head_dim + x];
            }
            Tdata l_curr = l_[l_offset + i * B_r + tx];

            // S_i^(j) = Q_i @ K_j^T * softmax_scale
            for (size_t y = 0; y < B_c; ++y) {
                // mask
                if (mask_ != nullptr && mask_[(i * B_r + tx) * seq_len_kv + j * B_c + y] == -INFINITY) {
                    s_i[tx * B_c + y] = -INFINITY;
                    continue;
                };

                Tdata sum = 0;
                for (size_t x = 0; x < head_dim; ++x) {
                    sum += q_i[tx * head_dim + x] * k_j[y * head_dim + x];
                }
                sum *= softmax_scale;
                s_i[tx * B_c + y] = sum;
            }

            // P_i^(j) = exp(S_ij - L_i)
            for (size_t y = 0; y < B_c; ++y) {
                if constexpr (std::is_same_v<Tdata, half> || std::is_same_v<Tdata, __nv_bfloat16>) {
                    s_i[tx * B_c + y] = hexp(s_i[tx * B_c + y] - l_curr);
                } else {
                    s_i[tx * B_c + y] = expf(s_i[tx * B_c + y] - l_curr);
                }
            }
            __syncthreads();

            // dV_j = dV_j + P_i^(j)^T @ dO_i
            for (size_t x = 0; x < head_dim; ++x) {
                Tdata sum = 0;
                for (size_t y = 0; y < B_r; ++y) {
                    sum += s_i[y * B_c + tx] * grad_out_i[y * head_dim + x];
                }
                grad_v_j[tx * head_dim + x] += sum;
            }

            // dP_i^(j) = dO_i @ V_j^T
            for (size_t y = 0; y < B_c; ++y) {
                Tdata sum = 0;
                for (size_t x = 0; x < head_dim; ++x) {
                    sum += grad_out_i[tx * head_dim + x] * v_j[y * head_dim + x];
                }
                grad_s_i[tx * B_c + y] = sum;
            }

            // dS_i^(j) = P_i^(j) * (dP_i^(j) - D_i)
            for (size_t y = 0; y < B_c; ++y) {
                grad_s_i[tx * B_c + y] = s_i[tx * B_c + y] * (grad_s_i[tx * B_c + y] - D_i);
            }

            // dQ_i = dQ_i + dS_i^(j) @ K_j
            for (size_t x = 0; x < head_dim; ++x) {
                Tdata sum = 0;
                for (size_t y = 0; y < B_c; ++y) {
                    sum += grad_s_i[tx * B_c + y] * k_j[y * head_dim + x];
                }
                sum *= softmax_scale;
                grad_q_[qo_offset + (i * B_r + tx) * qo_stride_s + x] += sum;
            }
            __syncthreads();

            // dK_j = dK_j + dS_i^(j)^T @ Q_i
            for (size_t x = 0; x < head_dim; ++x) {
                Tdata sum = 0;
                for (size_t y = 0; y < B_r; ++y) {
                    sum += grad_s_i[y * B_c + tx] * q_i[y * head_dim + x];
                }
                sum *= softmax_scale;
                grad_k_j[tx * head_dim + x] += sum;
            }
        }

        // write dK_j, dV_j to HBM
        for (size_t x = 0; x < head_dim; ++x) {
            size_t offset = bx * kv_stride_b * group + by * kv_stride_n;
            grad_k_[offset + (j * B_c + tx) * kv_stride_s * group + x] = grad_k_j[tx * head_dim + x];
            grad_v_[offset + (j * B_c + tx) * kv_stride_s * group + x] = grad_v_j[tx * head_dim + x];
        }
    }
}

#endif // __FLASH_ATTENTION_BACKWARD_KERNEL_CUH__
