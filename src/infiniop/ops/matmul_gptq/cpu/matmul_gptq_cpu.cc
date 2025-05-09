#include "matmul_gptq_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../handle.h"
#include <cassert>
#include <cblas.h>
#include <fstream>
#include <lapacke.h>

#ifdef NDEBUG
#define SAFE_ASSERT(x) ((void)(x))
#else
#define SAFE_ASSERT(x) assert(x)
#endif

namespace op::matmul_gptq::cpu {
Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t c_desc,
                                  infiniopTensorDescriptor_t a_desc,
                                  infiniopTensorDescriptor_t packed_weights_desc,
                                  infiniopTensorDescriptor_t b_scale_desc,
                                  infiniopTensorDescriptor_t zero_desc) {
    auto atype = a_desc->dtype();

    if ((atype != INFINI_DTYPE_F16 && atype != INFINI_DTYPE_BF16)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    int m = int(c_desc->dim(1));
    int n = int(c_desc->dim(0));
    int k = int(a_desc->dim(0));
    int num_groups = int(b_scale_desc->dim(1));
    int group_size = num_groups > 1 ? k / num_groups : -1;

    int blocksize = 128;
    size_t min_workspace_size = k * k * sizeof(float) + (2 * n * k + n * blocksize) * infiniSizeOf(atype);

    *desc_ptr = new Descriptor(m, n, k, min_workspace_size, atype, num_groups, group_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

float quantize(float x, float s, float z, int maxq) {
    int q = static_cast<int>(std::roundf(x / s + z));
    q = std::max(0, std::min(maxq, q));
    return s * (q - z);
}

template <typename T>
void find_params(T *x, T *b_scale, T *zero, int N, int K,
                 int bits = 4, bool sym = false, bool mse = false,
                 float norm = 2.4, int grid = 100, float maxshrink = 0.8) {
    float maxq = std::pow(2, bits) - 1;
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
        float x_min = __FLT_MAX__;
        float x_max = -__FLT_MAX__;
        for (int k = 0; k < K; k++) {
            if (utils::cast<float>(x[n * K + k]) < x_min) {
                x_min = utils::cast<float>(x[n * K + k]);
            }
            if (utils::cast<float>(x[n * K + k]) > x_max) {
                x_max = utils::cast<float>(x[n * K + k]);
            }
        }
        if (sym) {
            x_max = std::fmax(std::abs(x_min), x_max);
            if (x_min < 0) {
                x_min = -x_max;
            }
        }
        if (x_min == 0 && x_max == 0) {
            x_min = -1;
            x_max = 1;
        }
        if constexpr (std::is_same<T, fp16_t>::value) {
            b_scale[n] = utils::cast<fp16_t>((x_max - x_min) / maxq);
            if (sym) {
                zero[n] = utils::cast<fp16_t>((maxq + 1.0f) * 0.5f);
            } else {
                zero[n] = utils::cast<fp16_t>(-x_min * maxq / (x_max - x_min));
            }
        } else if constexpr (std::is_same<T, float>::value) {
            b_scale[n] = (x_max - x_min) / maxq;
            if (sym) {
                zero[n] = (maxq + 1) * 0.5;
            } else {
                zero[n] = -x_min / b_scale[n];
            }
        }
        if (mse) {
            float best = __FLT_MAX__;
            for (int i = 0; i < int(maxshrink * grid); i++) {
                float p = 1 - i / grid;
                float x_min_1 = p * x_min;
                float x_max_1 = p * x_max;
                float scale_1 = (x_max_1 - x_min_1) / maxq;
                float zero_1 = (sym ? utils::cast<float>(zero[n]) : std::roundf(-x_min_1 / scale_1));
                float err = 0.0f;
                for (int k = 0; k < K; k++) {
                    float q = quantize(utils::cast<float>(x[n * K + k]), scale_1, zero_1, maxq);
                    q -= utils::cast<float>(x[n * K + k]);
                    q = std::abs(q);
                    q = std::pow(q, norm);
                    err += q;
                }
                if (err < best) {
                    best = err;
                    if constexpr (std::is_same<T, fp16_t>::value) {
                        b_scale[n] = utils::cast<fp16_t>(scale_1);
                        zero[n] = utils::cast<fp16_t>(zero_1);
                    } else if constexpr (std::is_same<T, float>::value) {
                        b_scale[n] = scale_1;
                        zero[n] = zero_1;
                    }
                }
            }
        }
    }
}

template <typename T>
void add_batch(const T *inp, float *Hess, float nsamples, int M, int K) { // Hess, nsamples默认是0
    int tmp = 1;
#pragma omp parallel for
    for (int index = 0; index < K * K; index++) {
        Hess[index] = nsamples / (nsamples + tmp) * Hess[index];
    }
    nsamples += tmp;
#pragma omp parallel for
    for (int index = 0; index < K * K; index++) {
        int j = index % K;
        int i = index / K;
        float s = 0.0f;
        for (int m = 0; m < M; m++) {
            if constexpr (std::is_same<T, fp16_t>::value) {
                s += utils::cast<float>(inp[i * M + m]) * utils::cast<float>(inp[j * M + m]) * pow(nsamples / (nsamples + tmp), 2);
            } else if constexpr (std::is_same<T, float>::value) {
                s += inp[i * M + m] * inp[j * M + m] * pow(nsamples / (nsamples + tmp), 2);
            }
        }
        Hess[i * K + j] = s;
        if (i != j) {
            Hess[j * K + i] = s;
        }
    }
}

void cholesky(float *Hess, int K) {
    // Step 1: Cholesky分解，lower三角
    int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', K, Hess, K);
    SAFE_ASSERT(info == 0);

    // Step 2: 求逆，得到 (L * L^T)^{-1}
    info = LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', K, Hess, K);
    SAFE_ASSERT(info == 0);

    // Step 3: 手动补齐上三角
#pragma omp parallel for
    for (int i = 0; i < K; ++i) {
        for (int j = i + 1; j < K; ++j) {
            Hess[i * K + j] = Hess[j * K + i];
        }
    }

    // Step 4: 再次Cholesky分解，这次是upper三角
    info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'U', K, Hess, K);
    SAFE_ASSERT(info == 0);

    // Step 5: 清空下三角，只保留上三角
#pragma omp parallel for
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < i; ++j) {
            Hess[i * K + j] = 0.0f;
        }
    }
}

template <typename T>
void fasterquant(T *weight, T *Q, T *Err, T *b_scale, T *zero, float *Hess,
                 int M, int K, int N,
                 int blocksize = 128, float percdamp = 0.01, int group_size = -1,
                 int bits = 4, bool sym = false, bool mse = false,
                 float norm = 2.4, int grid = 100, float maxshrink = 0.8) {
    float maxq = std::pow(2, bits) - 1;
    int num_groups = (group_size == -1 ? 1 : K / group_size);

    if (group_size == -1) {
        find_params(weight, b_scale, zero, N, K, bits, sym, mse, norm, grid, maxshrink);
    }
    float damp = 0.0f;

#pragma omp parallel for reduction(+ : damp)
    for (int dead = 0; dead < K; ++dead) {
        bool condition = false;
        if (Hess[dead * K + dead] == 0.0f) {
            Hess[dead * K + dead] = 1.0f;
            condition = true;
        }
        damp += Hess[dead * K + dead];

        if (condition) {
            for (int n = 0; n < N; ++n) {
                if constexpr (std::is_same<T, fp16_t>::value) {
                    weight[n * K + dead] = utils::cast<fp16_t>(0.0f);
                } else if constexpr (std::is_same<T, float>::value) {
                    weight[n * K + dead] = 0.0f;
                }
            }
        }
    }

    damp = percdamp * damp / K;
#pragma omp parallel for
    for (int dead = 0; dead < K; dead++) {
        Hess[dead * K + dead] += damp;
    }
    cholesky(Hess, K);
    for (int index = 0; index < K / blocksize; index++) {
        for (int i = 0; i < blocksize; i++) {
            float d = Hess[(index * blocksize + i) * K + index * blocksize + i];

            if (group_size != -1) {
                if ((index * blocksize + i) % group_size == 0) {
                    int ind = (index * blocksize + i) / group_size;
                    for (int n = 0; n < N; n++) {
                        find_params(&weight[n * K + index * blocksize + i], &b_scale[n * num_groups + ind], &zero[n * num_groups + ind], 1, group_size, bits, sym, mse, norm, grid, maxshrink);
                    }
                }
            }
            int ind = (group_size != -1 ? (index * blocksize + i) / group_size : 0);
            for (int n = 0; n < N; n++) {
                float q = quantize(utils::cast<float>(weight[n * K + index * blocksize + i]), utils::cast<float>(b_scale[n * num_groups + ind]), utils::cast<float>(zero[n * num_groups + ind]), maxq);
                if constexpr (std::is_same<T, fp16_t>::value) {
                    Q[n * K + index * blocksize + i] = utils::cast<fp16_t>(q);
                } else if constexpr (std::is_same<T, float>::value) {
                    Q[n * K + index * blocksize + i] = q;
                }

                float w = utils::cast<float>(weight[n * K + index * blocksize + i]);
                float err = (w - q) / d;

                if (group_size == -1) {
                    for (int j = i; j < blocksize; j++) {
                        if constexpr (std::is_same<T, fp16_t>::value) {
                            weight[n * K + index * blocksize + j] = utils::cast<fp16_t>(utils::cast<float>(weight[n * K + index * blocksize + j]) - err * Hess[(index * blocksize + i) * K + j]);
                        } else if constexpr (std::is_same<T, float>::value) {
                            weight[n * K + index * blocksize + j] -= err * Hess[(index * blocksize + i) * K + j];
                        }
                    }
                }

                if constexpr (std::is_same<T, fp16_t>::value) {
                    Err[n * blocksize + i] = utils::cast<fp16_t>(err);
                } else if constexpr (std::is_same<T, float>::value) {
                    Err[n * blocksize + i] = err;
                }
            }
        }
        int i_2 = std::min((index + 1) * blocksize, K);
        for (int n = 0; n < N; n++) {
            for (int j = i_2; j < K; j++) {
                float s = 0.0f;
                for (int b = 0; b < blocksize; b++) {
                    s += utils::cast<float>(Err[n * blocksize + b]) * Hess[(index * blocksize + b) * K + j];
                }
                if constexpr (std::is_same<T, fp16_t>::value) {
                    weight[n * K + j] = utils::cast<fp16_t>(utils::cast<float>(weight[n * K + j]) - s);
                } else if constexpr (std::is_same<T, float>::value) {
                    weight[n * K + j] -= s;
                }
            }
        }
    }
}

void PackQuantizedWeight(fp16_t *Q, fp16_t *b_scale, fp16_t *zero,
                         int32_t *packed_weight, int K, int N, int group_size, int bits = 4) {
    int maxq = int(std::pow(2, bits) - 1);
    int num_groups = (group_size == -1) ? 1 : K / group_size;
    int blocks_per_group = (group_size == -1) ? K / 8 : group_size / 8;

#pragma omp parallel for
    for (int index = 0; index < N * num_groups * blocks_per_group; ++index) {
        int n = index / (num_groups * blocks_per_group);
        int rem = index % (num_groups * blocks_per_group);
        int g = rem / blocks_per_group;
        int b = rem % blocks_per_group;

        float scale = utils::cast<float>(b_scale[n * num_groups + g]);
        float zero_f = utils::cast<float>(zero[n * num_groups + g]);

        int row_base = (group_size == -1) ? b * 8 : g * group_size + b * 8;
        int row_block_idx = row_base / 8;

        int32_t packed = 0;
        for (int i = 0; i < 8; ++i) {
            int k = row_base + i;
            float val = utils::cast<float>(Q[n * K + k]); // Q: [N, K]
            int q = static_cast<int>(std::roundf(val / scale + zero_f));
            q = std::max(0, std::min(maxq, q)); // clamp to [0, maxq]
            packed |= (q & 0xF) << (i * 4);
        }

        packed_weight[n * (K / 8) + row_block_idx] = packed;
    }
}

void MatmulPackedWeight(fp16_t *C, const fp16_t *A, int32_t *packed_weight,
                        fp16_t *b_scale, fp16_t *zero,
                        int M, int K, int N, int group_size) {
    int num_groups = (group_size == -1) ? 1 : K / group_size;
    int blocks_per_group = (group_size == -1) ? K / 8 : group_size / 8;
#pragma omp parallel for
    for (int index = 0; index < N * M; index++) {
        int m = index % M;
        int n = index / M;
        float acc = 0.0f;

        for (int g = 0; g < num_groups; ++g) {
            float scale = utils::cast<float>(b_scale[n * num_groups + g]);
            float zero_f = utils::cast<float>(zero[n * num_groups + g]);

            for (int b = 0; b < blocks_per_group; ++b) {
                int row_base = (group_size == -1) ? b * 8 : g * group_size + b * 8;
                int row_block_idx = row_base / 8;
                int32_t packed = packed_weight[n * (K / 8) + row_block_idx];

                for (int i = 0; i < 8; ++i) {
                    int k = row_base + i;
                    int q = (packed >> (i * 4)) & 0xF;
                    float w = (q - zero_f) * scale;

                    float a_val = utils::cast<float>(A[k * M + m]); // A: [K, M]
                    acc += w * a_val;
                }
            }
        }

        C[index] = utils::cast<fp16_t>(acc);
    }
}

void quantWeights(void *workspace, int32_t *packed_weights,
                  fp16_t *b_scale,
                  fp16_t *zero,
                  const fp16_t *A,
                  const fp16_t *B,
                  int M, int K, int N,
                  int group_size) {

    int blocksize = 128;
    float percdamp = 0.01;

    int bits = 4;
    bool sym = false;
    bool mse = false;
    float norm = 2.4;
    int grid = 100;
    float maxshrink = 0.8;
    float nsamples = 0.0f;

    char *tmp = (char *)workspace + K * K * sizeof(float);
    float *Hess = (float *)workspace; //[K, K]
    fp16_t *Q = (fp16_t *)tmp;        //[N, K]
    fp16_t *weight = Q + N * K;       //[N, K]
    fp16_t *Err = weight + N * K;     //[N, blocksize=128]
    memset(Hess, 0, sizeof(float) * K * K);
    memcpy(weight, B, N * K * sizeof(fp16_t));
    add_batch<fp16_t>(A, Hess, nsamples, M, K);
    fasterquant<fp16_t>(weight, Q, Err, b_scale, zero, Hess,
                        M, K, N,
                        blocksize, percdamp, group_size,
                        bits, sym, mse,
                        norm, grid, maxshrink);
    PackQuantizedWeight(Q, b_scale, zero, packed_weights, K, N, group_size, bits);
}

void caculate(void *workspace, fp16_t *C, const fp16_t *A,
              int32_t *packed_weights, fp16_t *b_scale, fp16_t *zero,
              int M, int K, int N, int group_size) {

    MatmulPackedWeight(C, A, packed_weights, b_scale, zero, M, K, N, group_size);
}

infiniStatus_t Descriptor::quant(
    void *workspace,
    size_t workspace_size,
    void *packed_weights,
    void *b_scale,
    void *zero,
    const void *a,
    const void *b,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (_atype == INFINI_DTYPE_F16) {
        quantWeights(workspace, (int32_t *)packed_weights,
                     (fp16_t *)b_scale,
                     (fp16_t *)zero,
                     (fp16_t *)a, (fp16_t *)b, _m, _k, _n, _group_size);

    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    void *packed_weights,
    void *b_scale,
    void *zero,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (_atype == INFINI_DTYPE_F16) {
        caculate(workspace, (fp16_t *)c, (fp16_t *)a, (int32_t *)packed_weights, (fp16_t *)b_scale, (fp16_t *)zero,
                 _m, _k, _n, _group_size);

    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::matmul_gptq::cpu
