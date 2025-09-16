#ifndef __FLASH_ATTENTION_KERNEL_CUH__
#define __FLASH_ATTENTION_KERNEL_CUH__

template <typename Tdata>
__device__ void flashAttentionBlock(
    Tdata *out_, Tdata *l_,
    const Tdata *q_, const Tdata *k_, const Tdata *v_, const float *mask_,
    const size_t seq_len_q, const size_t seq_len_kv,
    const size_t head_dim, const size_t group,
    const size_t B_r, const size_t B_c, const size_t T_r, const size_t T_c,
    const Tdata softmax_scale,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n,
    ptrdiff_t l_stride_b, ptrdiff_t l_stride_s, ptrdiff_t l_stride_n) {

    size_t bx = blockIdx.x;  // batch                          -> batch_size
    size_t by = blockIdx.y;  // q's head index                 -> num_heads_q
    size_t tx = threadIdx.x; // q's row index within one block -> B_r/B_c

    size_t qo_offset = bx * qo_stride_b + by * qo_stride_n;
    size_t kv_offset = bx * kv_stride_b + by / group * kv_stride_n;
    size_t l_offset = bx * l_stride_b + by * seq_len_q;

    extern __shared__ __align__(sizeof(Tdata)) char shared_mem[];
    Tdata *q_i = reinterpret_cast<Tdata *>(shared_mem);
    Tdata *k_j = reinterpret_cast<Tdata *>(q_i + B_r * head_dim);
    Tdata *v_j = reinterpret_cast<Tdata *>(k_j + B_c * head_dim);
    Tdata *s_i = reinterpret_cast<Tdata *>(v_j + B_c * head_dim);

    for (size_t i = 0; i < T_r; ++i) {
        // skip when over q's seq_len
        if (i * B_r + tx >= seq_len_q) {
            break;
        }

        // load q_i from HBM to on-chip SRAM
        for (size_t x = 0; x < head_dim; ++x) {
            q_i[tx * head_dim + x] = q_[qo_offset + (i * B_r + tx) * qo_stride_s + x];
        }
        // initial m, l
        Tdata row_m_prev = -INFINITY;
        Tdata row_l_prev = 0;

        for (size_t j = 0; j < T_c; ++j) {
            __syncthreads();
            // load k_j, v_j from HBM to on-chip SRAM
            for (size_t y = 0; y < B_c; ++y) {
                for (size_t x = 0; x < head_dim; ++x) {
                    k_j[y * head_dim + x] = k_[kv_offset + (y + j * B_c) * kv_stride_s + x];
                    v_j[y * head_dim + x] = v_[kv_offset + (y + j * B_c) * kv_stride_s + x];
                }
            }
            __syncthreads();

            Tdata row_m = -INFINITY;
            for (size_t y = 0; y < B_c; ++y) {
                if (j * B_c + y >= seq_len_kv) {
                    break;
                }

                // mask
                if (mask_ != nullptr && mask_[(i * B_r + tx) * seq_len_kv + j * B_c + y] == -INFINITY) {
                    s_i[tx * B_c + y] = -INFINITY;
                    continue;
                };

                // S_i^(j) = Q_i @ K_j^T / softmax_scale
                Tdata sum = 0;
                for (size_t x = 0; x < head_dim; ++x) {
                    sum += q_i[tx * head_dim + x] * k_j[y * head_dim + x];
                }
                sum *= softmax_scale;

                s_i[tx * B_c + y] = sum;

                if constexpr (std::is_same_v<Tdata, half> || std::is_same_v<Tdata, __hpcc_bfloat16>) {
                    row_m = __hmax(row_m, sum);
                } else {
                    row_m = fmaxf(row_m, sum);
                }
            }

            // m_i^(j) = max(m_i^(j - 1), rowmax(S_i^(j)))
            Tdata new_row_m;
            if constexpr (std::is_same_v<Tdata, half> || std::is_same_v<Tdata, __hpcc_bfloat16>) {
                new_row_m = __hmax(row_m_prev, row_m);
            } else {
                new_row_m = fmaxf(row_m_prev, row_m);
            }

            // rowsum(P_i^(j))
            Tdata row_l = 0;
            for (size_t y = 0; y < B_r; ++y) {
                if (j * B_c + y >= seq_len_kv) {
                    break;
                }

                // mask
                if (mask_ != nullptr && mask_[(i * B_r + tx) * seq_len_kv + j * B_c + y] == -INFINITY) {
                    continue;
                }

                // P_i^(j) = exp(S_i^(j) - m_i^(j))
                if constexpr (std::is_same_v<Tdata, half> || std::is_same_v<Tdata, __hpcc_bfloat16>) {
                    if (__hisinf(new_row_m)) {
                        s_i[tx * B_c + y] = 1.0;
                    } else {
                        s_i[tx * B_c + y] = hexp(s_i[tx * B_c + y] - new_row_m);
                    }
                } else {
                    if (isinf(new_row_m)) {
                        s_i[tx * B_c + y] = 1.0;
                    } else {
                        s_i[tx * B_c + y] = expf(s_i[tx * B_c + y] - new_row_m);
                    }
                }

                row_l += s_i[tx * B_c + y];
            }

            // l_i^(j) = exp(m_i^(j - 1) - m_i^(j - 1)) * l_i^(j - 1) + rowsum(P_i^(j))
            Tdata row_m_exp;
            if constexpr (std::is_same_v<Tdata, half> || std::is_same_v<Tdata, __hpcc_bfloat16>) {
                if (__hisinf(row_m_prev)) {
                    row_m_exp = 1.0;
                } else {
                    row_m_exp = hexp(row_m_prev - new_row_m);
                }
            } else {
                if (isinf(new_row_m)) {
                    row_m_exp = 1.0;
                } else {
                    row_m_exp = expf(row_m_prev - new_row_m);
                }
            }
            Tdata new_row_l = (row_m_exp * row_l_prev) + row_l;

            // out_i^(j) = diag(exp(m_i^(j - 1) - m_i^(y))) * O_i^(j - 1) + P_i^(j) * V_j
            for (size_t x = 0; x < head_dim; ++x) {
                Tdata pv = 0;
                for (size_t y = 0; y < B_c; ++y) {
                    if (j * B_c + y >= seq_len_kv) {
                        break;
                    }

                    // mask
                    if (mask_ != nullptr && mask_[(i * B_r + tx) * seq_len_kv + j * B_c + y] == -INFINITY) {
                        continue;
                    }

                    pv += s_i[tx * B_c + y] * v_j[y * head_dim + x];
                }

                out_[qo_offset + (i * B_r + tx) * qo_stride_s + x] = row_m_exp * out_[qo_offset + (i * B_r + tx) * qo_stride_s + x] + pv;
            }

            row_m_prev = new_row_m;
            row_l_prev = new_row_l;
        }

        // O_i = O_i^(Tc) / l_i^(Tc)
        for (size_t x = 0; x < head_dim; ++x) {
            out_[qo_offset + (i * B_r + tx) * qo_stride_s + x] /= row_l_prev;
        }

        // L_i = m_i^(Tc) + log(l_i^(Tc))
        if constexpr (std::is_same_v<Tdata, half> || std::is_same_v<Tdata, __hpcc_bfloat16>) {
            l_[l_offset + i * B_r + tx] = row_m_prev + hlog(row_l_prev);
        } else {
            l_[l_offset + i * B_r + tx] = row_m_prev + logf(row_l_prev);
        }
    }
}

#endif // __FLASH_ATTENTION_KERNEL_CUH__
