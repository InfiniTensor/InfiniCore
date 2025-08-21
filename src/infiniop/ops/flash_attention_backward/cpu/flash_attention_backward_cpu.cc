#include "flash_attention_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "math.h"

namespace op::flash_attention_backward::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t grad_q_desc,
    infiniopTensorDescriptor_t grad_k_desc,
    infiniopTensorDescriptor_t grad_v_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t grad_out_desc,
    infiniopTensorDescriptor_t mask_desc,
    infiniopAttentionMaskType_t mask_type) {

    auto info = FlashAttentionBackwardInfo::create(
        grad_q_desc, grad_k_desc, grad_v_desc,
        q_desc, k_desc, v_desc,
        grad_out_desc, mask_desc, mask_type);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t flashAttention(
    T *out, T *l, const T *q, const T *k, const T *v, const float *mask,
    size_t batch_size,
    size_t nums_head_q, size_t nums_head_kv,
    size_t seq_len_q, size_t seq_len_kv,
    size_t head_dim, size_t group,
    size_t B_r, size_t B_c, size_t T_r, size_t T_c,
    size_t qo_stride_b, size_t qo_stride_s, size_t qo_stride_n,
    size_t kv_stride_b, size_t kv_stride_s, size_t kv_stride_n,
    size_t l_stride_b, size_t l_stride_s, size_t l_stride_n) {

    std::memset(out, 0, batch_size * nums_head_q * seq_len_q * head_dim * sizeof(T));
    std::memset(l, 0, batch_size * nums_head_q * seq_len_q * sizeof(T));

    float softmax_scale = 1.f / sqrt(float(head_dim));

#pragma omp parallel for
    for (ptrdiff_t bx = 0; bx < ptrdiff_t(batch_size); ++bx) {
        for (size_t by = 0; by < nums_head_q; ++by) {
            size_t qo_offset = bx * qo_stride_b + by * qo_stride_n;
            size_t kv_offset = bx * kv_stride_b + by / group * kv_stride_n;
            size_t l_offset = bx * l_stride_b + by * seq_len_q;

            std::vector<float> q_i(B_r * head_dim);
            std::vector<float> k_j(B_c * head_dim);
            std::vector<float> v_j(B_c * head_dim);
            std::vector<float> s_i(B_r * B_c);

            for (size_t i = 0; i < T_r; ++i) {
                for (size_t tx = 0; tx < B_r; ++tx) {
                    // skip when over q's seq_len
                    if (i * B_r + tx >= seq_len_q) {
                        break;
                    }

                    // load q_i
                    for (size_t x = 0; x < head_dim; ++x) {
                        q_i[tx * head_dim + x] = utils::cast<float>(q[qo_offset + (i * B_r + tx) * qo_stride_s + x]);
                    }

                    // initial m, l
                    float row_m_prev = -INFINITY;
                    float row_l_prev = 0;

                    for (size_t j = 0; j < T_c; ++j) {
                        // load k_j, v_j
                        for (size_t y = 0; y < B_c; ++y) {
                            for (size_t x = 0; x < head_dim; ++x) {
                                k_j[y * head_dim + x] = utils::cast<float>(k[kv_offset + (y + j * B_c) * kv_stride_s + x]);
                                v_j[y * head_dim + x] = utils::cast<float>(v[kv_offset + (y + j * B_c) * kv_stride_s + x]);
                            }
                        }

                        float row_m = -INFINITY;
                        for (size_t y = 0; y < B_c; ++y) {
                            if (j * B_c + y >= seq_len_kv) {
                                break;
                            }

                            // mask
                            if (mask != nullptr && mask[(i * B_r + tx) * seq_len_kv + j * B_c + y] == -INFINITY) {
                                s_i[tx * B_c + y] = -INFINITY;
                                continue;
                            }

                            // S_i^(j) = Q_i @ K_j^T / softmax_scale
                            float sum = 0;
                            for (size_t x = 0; x < head_dim; ++x) {
                                sum += q_i[tx * head_dim + x] * k_j[y * head_dim + x];
                            }
                            sum *= softmax_scale;

                            s_i[tx * B_c + y] = sum;

                            row_m = std::max(row_m, sum);
                        }

                        // m_i^(j) = max(m_i^(j - 1), rowmax(S_i^(j)))
                        float new_row_m = std::max(row_m_prev, row_m);

                        // rowsum(P_i^(j))
                        float row_l = 0;
                        for (size_t y = 0; y < B_r; ++y) {
                            if (j * B_c + y >= seq_len_kv) {
                                break;
                            }

                            // mask
                            if (mask != nullptr && mask[(i * B_r + tx) * seq_len_kv + j * B_c + y] == -INFINITY) {
                                continue;
                            }

                            // P_i^(j) = exp(S_i^(j) - m_i^(j))
                            if (new_row_m == -INFINITY) {
                                s_i[tx * B_c + y] = 1.0;
                            } else {
                                s_i[tx * B_c + y] = exp(s_i[tx * B_c + y] - new_row_m);
                            }

                            row_l += s_i[tx * B_c + y];
                        }

                        // l_i^(j) = exp(m_i^(j - 1) - m_i^(j - 1)) * l_i^(j - 1) + rowsum(P_i^(j))
                        float row_m_exp;
                        if (row_m_prev == -INFINITY) {
                            row_m_exp = 1.0;
                        } else {
                            row_m_exp = exp(row_m_prev - new_row_m);
                        }
                        float new_row_l = (row_m_exp * row_l_prev) + row_l;

                        // out_i^(j) = diag(exp(m_i^(j - 1) - m_i^(y))) * O_i^(j - 1) + P_i^(j) * V_j
                        for (size_t x = 0; x < head_dim; ++x) {
                            float pv = 0;
                            for (size_t y = 0; y < B_c; ++y) {
                                if (j * B_c + y >= seq_len_kv) {
                                    break;
                                }

                                // mask
                                if (mask != nullptr && mask[(i * B_r + tx) * seq_len_kv + j * B_c + y] == -INFINITY) {
                                    continue;
                                }

                                pv += s_i[tx * B_c + y] * v_j[y * head_dim + x];
                            }

                            out[qo_offset + (i * B_r + tx) * qo_stride_s + x] = utils::cast<T>(row_m_exp * utils::cast<float>(out[qo_offset + (i * B_r + tx) * qo_stride_s + x]) + pv);
                        }

                        row_m_prev = new_row_m;
                        row_l_prev = new_row_l;
                    }

                    // O_i = O_i^(Tc) / l_i^(Tc)
                    for (size_t x = 0; x < head_dim; ++x) {
                        out[qo_offset + (i * B_r + tx) * qo_stride_s + x] = utils::cast<T>(utils::cast<float>(out[qo_offset + (i * B_r + tx) * qo_stride_s + x]) / row_l_prev);
                    }

                    l[l_offset + i * B_r + tx] = utils::cast<T>(row_m_prev + log(row_l_prev));
                }
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t flashAttentionBackward(
    T *grad_q, T *grad_k, T *grad_v,
    const T *q, const T *k, const T *v, const T *out, const T *grad_out, const T *l,
    const float *mask,
    size_t batch_size,
    size_t nums_head_q, size_t nums_head_kv,
    size_t seq_len_q, size_t seq_len_kv,
    size_t head_dim, size_t group,
    size_t B_r, size_t B_c, size_t T_r, size_t T_c,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n,
    ptrdiff_t l_stride_b, ptrdiff_t l_stride_s, ptrdiff_t l_stride_n) {

    std::memset(grad_q, 0, batch_size * nums_head_q * seq_len_q * head_dim * sizeof(T));
    std::memset(grad_k, 0, batch_size * nums_head_kv * seq_len_kv * head_dim * sizeof(T));
    std::memset(grad_v, 0, batch_size * nums_head_kv * seq_len_kv * head_dim * sizeof(T));

    float softmax_scale = 1.f / sqrt(float(head_dim));

#pragma omp parallel for
    for (ptrdiff_t bx = 0; bx < ptrdiff_t(batch_size); ++bx) {
        for (size_t by = 0; by < nums_head_q; ++by) {
            size_t qo_offset = bx * qo_stride_b + by * qo_stride_n;
            size_t kv_offset = bx * kv_stride_b + by / group * kv_stride_n;
            size_t l_offset = bx * l_stride_b + by * seq_len_q;

            std::vector<float> q_i(B_r * head_dim);
            std::vector<float> out_i(B_r * head_dim);
            std::vector<float> grad_out_i(B_r * head_dim);

            std::vector<float> k_j(B_c * head_dim);
            std::vector<float> v_j(B_c * head_dim);
            std::vector<float> grad_k_j(B_c * head_dim);
            std::vector<float> grad_v_j(B_c * head_dim);

            std::vector<float> s_i(B_r * B_c);
            std::vector<float> grad_s_i(B_r * B_c);

            std::vector<float> D_i(B_r);

            for (size_t j = 0; j < T_c; ++j) {
                for (size_t tx = 0; tx < B_r; ++tx) {
                    // load k_j, v_j and initialize grad_k_j, grad_q_j to 0
                    for (size_t x = 0; x < head_dim; ++x) {
                        k_j[tx * head_dim + x] = utils::cast<float>(k[kv_offset + (j * B_c + tx) * kv_stride_s + x]);
                        v_j[tx * head_dim + x] = utils::cast<float>(v[kv_offset + (j * B_c + tx) * kv_stride_s + x]);
                        grad_k_j[tx * head_dim + x] = 0;
                        grad_v_j[tx * head_dim + x] = 0;
                    }
                }

                for (size_t i = 0; i < T_r; ++i) {
                    for (size_t tx = 0; tx < B_r; ++tx) {
                        // load q_i, out_i, grad_out_i
                        D_i[tx] = 0;
                        for (size_t x = 0; x < head_dim; ++x) {
                            q_i[tx * head_dim + x] = utils::cast<float>(q[qo_offset + (i * B_r + tx) * qo_stride_s + x]);
                            out_i[tx * head_dim + x] = utils::cast<float>(out[qo_offset + (i * B_r + tx) * qo_stride_s + x]);
                            grad_out_i[tx * head_dim + x] = utils::cast<float>(grad_out[qo_offset + (i * B_r + tx) * qo_stride_s + x]);
                            D_i[tx] += grad_out_i[tx * head_dim + x] * out_i[tx * head_dim + x];
                        }
                        float l_curr = utils::cast<float>(l[l_offset + i * B_r + tx]);

                        // S_i^(j) = Q_i @ K_j^T * softmax_scale
                        for (size_t y = 0; y < B_c; ++y) {
                            // mask
                            if (mask != nullptr && mask[(i * B_r + tx) * seq_len_kv + j * B_c + y] == -INFINITY) {
                                s_i[tx * B_c + y] = -INFINITY;
                                continue;
                            };

                            float sum = 0;
                            for (size_t x = 0; x < head_dim; ++x) {
                                sum += q_i[tx * head_dim + x] * k_j[y * head_dim + x];
                            }
                            sum *= softmax_scale;
                            s_i[tx * B_c + y] = sum;
                        }

                        // P_i^(j) = exp(S_ij - L_i)
                        for (size_t y = 0; y < B_c; ++y) {
                            s_i[tx * B_c + y] = exp(s_i[tx * B_c + y] - l_curr);
                        }
                    }

                    for (size_t tx = 0; tx < B_r; ++tx) {
                        // dV_j = dV_j + P_i^(j)^T @ dO_i
                        for (size_t x = 0; x < head_dim; ++x) {
                            float sum = 0;
                            for (size_t y = 0; y < B_r; ++y) {
                                sum += s_i[y * B_c + tx] * grad_out_i[y * head_dim + x];
                            }
                            grad_v_j[tx * head_dim + x] += sum;
                        }

                        // dP_i^(j) = dO_i @ V_j^T
                        for (size_t y = 0; y < B_c; ++y) {
                            float sum = 0;
                            for (size_t x = 0; x < head_dim; ++x) {
                                sum += grad_out_i[tx * head_dim + x] * v_j[y * head_dim + x];
                            }
                            grad_s_i[tx * B_c + y] = sum;
                        }

                        // dS_i^(j) = P_i^(j) * (dP_i^(j) - D_i)
                        for (size_t y = 0; y < B_c; ++y) {
                            grad_s_i[tx * B_c + y] = s_i[tx * B_c + y] * (grad_s_i[tx * B_c + y] - D_i[tx]);
                        }
                    }

                    for (size_t tx = 0; tx < B_r; ++tx) {
                        // dQ_i = dQ_i + dS_i^(j) @ K_j
                        for (size_t x = 0; x < head_dim; ++x) {
                            float sum = 0;
                            for (size_t y = 0; y < B_c; ++y) {
                                sum += grad_s_i[tx * B_c + y] * k_j[y * head_dim + x];
                            }
                            sum *= softmax_scale;
                            grad_q[qo_offset + (i * B_r + tx) * qo_stride_s + x] = utils::cast<T>(utils::cast<float>(grad_q[qo_offset + (i * B_r + tx) * qo_stride_s + x]) + sum);
                        }

                        // dK_j = dK_j + dS_i^(j)^T @ Q_i
                        for (size_t x = 0; x < head_dim; ++x) {
                            float sum = 0;
                            for (size_t y = 0; y < B_r; ++y) {
                                sum += grad_s_i[y * B_c + tx] * q_i[y * head_dim + x];
                            }
                            sum *= softmax_scale;
                            grad_k_j[tx * head_dim + x] += sum;
                        }
                    }
                }

                for (size_t tx = 0; tx < B_r; ++tx) {
                    // write dK_j, dV_j
                    for (size_t x = 0; x < head_dim; ++x) {
                        // size_t offset = bx * kv_stride_b * group + by * kv_stride_n;
                        // grad_k[offset + (j * B_c + tx) * kv_stride_s * group + x] = utils::cast<T>(grad_k_j[tx * head_dim + x]);
                        // grad_v[offset + (j * B_c + tx) * kv_stride_s * group + x] = utils::cast<T>(grad_v_j[tx * head_dim + x]);
                        grad_k[kv_offset + (j * B_c + tx) * kv_stride_s + x] = utils::cast<T>(utils::cast<float>(grad_k[kv_offset + (j * B_c + tx) * kv_stride_s + x]) + grad_k_j[tx * head_dim + x]);
                        grad_v[kv_offset + (j * B_c + tx) * kv_stride_s + x] = utils::cast<T>(utils::cast<float>(grad_v[kv_offset + (j * B_c + tx) * kv_stride_s + x]) + grad_v_j[tx * head_dim + x]);
                    }
                }
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *grad_q, void *grad_k, void *grad_v,
    const void *q, const void *k, const void *v, const void *grad_out,
    const void *mask,
    void *stream) const {

    size_t B_r = 2;
    size_t B_c = 2;

    size_t batch_size = _info.batch_size;
    size_t seq_len_q = _info.seq_len_q;
    size_t seq_len_kv = _info.seq_len_kv;
    size_t nums_head_q = _info.num_heads_q;
    size_t nums_head_kv = _info.num_heads_kv;
    size_t group = nums_head_q / nums_head_kv;
    size_t head_dim = _info.head_dim;

    const void *mask_input = nullptr;
    if (_info.is_masked) {
        if (_info.mask != nullptr) {
            mask_input = _info.mask;
        } else {
            mask_input = mask;
        }
    }

    size_t T_r = CEIL_DIV(seq_len_q, B_r);
    size_t T_c = CEIL_DIV(seq_len_kv, B_c);

    if (_info.dtype == INFINI_DTYPE_F32) {
        float *out = new float[batch_size * nums_head_q * seq_len_q * head_dim];
        float *l = new float[batch_size * nums_head_q * seq_len_q];

        CHECK_STATUS(flashAttention(
            (float *)out, (float *)l, (const float *)q, (const float *)k, (const float *)v,
            (const float *)mask_input,
            batch_size,
            nums_head_q, nums_head_kv,
            seq_len_q, seq_len_kv,
            head_dim, group,
            B_r, B_c, T_r, T_c,
            _info.qo_stride_b, _info.qo_stride_s, _info.qo_stride_n,
            _info.kv_stride_b, _info.kv_stride_s, _info.kv_stride_n,
            _info.l_stride_b, _info.l_stride_s, _info.l_stride_n));

        CHECK_STATUS(flashAttentionBackward(
            (float *)grad_q, (float *)grad_k, (float *)grad_v,
            (const float *)q, (const float *)k, (const float *)v, (const float *)out, (const float *)grad_out, (const float *)l,
            (const float *)mask_input,
            batch_size,
            nums_head_q, nums_head_kv,
            seq_len_q, seq_len_kv,
            head_dim, group,
            B_r, B_c, T_r, T_c,
            _info.qo_stride_b, _info.qo_stride_s, _info.qo_stride_n,
            _info.kv_stride_b, _info.kv_stride_s, _info.kv_stride_n,
            _info.l_stride_b, _info.l_stride_s, _info.l_stride_n));

    } else if (_info.dtype == INFINI_DTYPE_F16) {
        fp16_t *out = new fp16_t[batch_size * nums_head_q * seq_len_q * head_dim];
        fp16_t *l = new fp16_t[batch_size * nums_head_q * seq_len_q];

        CHECK_STATUS(flashAttention(
            (fp16_t *)out, (fp16_t *)l, (fp16_t *)q, (fp16_t *)k, (fp16_t *)v,
            (float *)mask_input,
            batch_size,
            nums_head_q, nums_head_kv,
            seq_len_q, seq_len_kv,
            head_dim, group,
            B_r, B_c, T_r, T_c,
            _info.qo_stride_b, _info.qo_stride_s, _info.qo_stride_n,
            _info.kv_stride_b, _info.kv_stride_s, _info.kv_stride_n,
            _info.l_stride_b, _info.l_stride_s, _info.l_stride_n));

        CHECK_STATUS(flashAttentionBackward(
            (fp16_t *)grad_q, (fp16_t *)grad_k, (fp16_t *)grad_v,
            (fp16_t *)q, (fp16_t *)k, (fp16_t *)v, (fp16_t *)out, (fp16_t *)grad_out, (fp16_t *)l,
            (float *)mask_input,
            batch_size,
            nums_head_q, nums_head_kv,
            seq_len_q, seq_len_kv,
            head_dim, group,
            B_r, B_c, T_r, T_c,
            _info.qo_stride_b, _info.qo_stride_s, _info.qo_stride_n,
            _info.kv_stride_b, _info.kv_stride_s, _info.kv_stride_n,
            _info.l_stride_b, _info.l_stride_s, _info.l_stride_n));

    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        bf16_t *out = new bf16_t[batch_size * nums_head_q * seq_len_q * head_dim];
        bf16_t *l = new bf16_t[batch_size * nums_head_q * seq_len_q];

        CHECK_STATUS(flashAttention(
            (bf16_t *)out, (bf16_t *)l, (bf16_t *)q, (bf16_t *)k, (bf16_t *)v,
            (float *)mask_input,
            batch_size,
            nums_head_q, nums_head_kv,
            seq_len_q, seq_len_kv,
            head_dim, group,
            B_r, B_c, T_r, T_c,
            _info.qo_stride_b, _info.qo_stride_s, _info.qo_stride_n,
            _info.kv_stride_b, _info.kv_stride_s, _info.kv_stride_n,
            _info.l_stride_b, _info.l_stride_s, _info.l_stride_n));

        CHECK_STATUS(flashAttentionBackward(
            (bf16_t *)grad_q, (bf16_t *)grad_k, (bf16_t *)grad_v,
            (bf16_t *)q, (bf16_t *)k, (bf16_t *)v, (bf16_t *)out, (bf16_t *)grad_out, (bf16_t *)l,
            (float *)mask_input,
            batch_size,
            nums_head_q, nums_head_kv,
            seq_len_q, seq_len_kv,
            head_dim, group,
            B_r, B_c, T_r, T_c,
            _info.qo_stride_b, _info.qo_stride_s, _info.qo_stride_n,
            _info.kv_stride_b, _info.kv_stride_s, _info.kv_stride_n,
            _info.l_stride_b, _info.l_stride_s, _info.l_stride_n));
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::flash_attention_backward::cpu
