#include "quantize_gptq_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../handle.h"
#include <cassert>
// 防止 __C 与 AVX512 冲突
// 判断是否为 ARM 架构（如 ARM64/AArch64）
#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
// 判断是否为 x86/x86_64 架构
#elif defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
#pragma push_macro("__C")
#undef __C
#include <immintrin.h>
#pragma pop_macro("__C")
#else
#error "Unsupported architecture: Neither ARM nor x86 detected."
#endif
#include <cfloat>
#ifdef NDEBUG
#define SAFE_ASSERT(x) ((void)(x))
#else
#define SAFE_ASSERT(x) assert(x)
#endif

namespace op::quantize_gptq::cpu {
Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(infiniopHandle_t handle_, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t c_desc,
                                  infiniopTensorDescriptor_t a_desc,
                                  infiniopTensorDescriptor_t packed_weights_desc,
                                  infiniopTensorDescriptor_t b_scale_desc,
                                  infiniopTensorDescriptor_t zero_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = MatmulGptqInfo::createMatmulGptqInfo(c_desc, a_desc, packed_weights_desc, b_scale_desc, zero_desc);
    CHECK_RESULT(result);
    MatmulGptqInfo info = result.take();
    size_t min_workspace_size
        = (info.k * info.k + info.n * info.block_size) * sizeof(float) + (2 * info.n * info.k) * infiniSizeOf(info.atype);

    *desc_ptr = new Descriptor(info, nullptr, min_workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

float quantize(float x, float s, float z, float minq, float maxq) {
    float q = std::roundf(x / s + z);
    q = std::max(minq, std::min(maxq, q));
    return s * (q - z);
}

template <typename T>
void find_params(T *x, T *b_scale, T *zero, int N, int K,
                 int bits = 4, bool sym = false, bool mse = false,
                 float norm = 2.4f, int grid = 100, float maxshrink = 0.8f, bool sign_ed = false) {
    float maxq;
    float minq;
    if (sign_ed) { // 如果有符号量化
        maxq = static_cast<float>(std::pow(2, bits - 1) - 1);
        minq = -static_cast<float>(std::pow(2, bits - 1));
    } else {
        maxq = static_cast<float>(std::pow(2, bits) - 1);
        minq = 0.0f;
    }
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
        float x_min = FLT_MAX;
        float x_max = -FLT_MAX;
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
            b_scale[n] = utils::cast<fp16_t>((x_max - x_min) / (maxq - minq));
            if (sym) {
                zero[n] = utils::cast<fp16_t>((maxq + minq + 1.0f) * 0.5f);
            } else {
                zero[n] = utils::cast<fp16_t>(-x_min * (maxq - minq) / (x_max - x_min));
            }
        } else if constexpr (std::is_same<T, float>::value) {
            b_scale[n] = (x_max - x_min) / (maxq - minq);
            if (sym) {
                zero[n] = (maxq + minq + 1.0f) * 0.5f;
            } else {
                zero[n] = -x_min / b_scale[n];
            }
        }
        if (mse) {
            float best = FLT_MAX;
            for (int i = 0; i < int(maxshrink * grid); i++) {
                float p = 1 - static_cast<float>(i) / static_cast<float>(grid);
                float x_min_1 = p * x_min;
                float x_max_1 = p * x_max;
                float scale_1 = (x_max_1 - x_min_1) / (maxq - minq);
                float zero_1 = (sym ? utils::cast<float>(zero[n]) : std::roundf(-x_min_1 / scale_1));
                float err = 0.0f;
                for (int k = 0; k < K; k++) {
                    float q = quantize(utils::cast<float>(x[n * K + k]), scale_1, zero_1, minq, maxq);
                    q -= utils::cast<float>(x[n * K + k]);
                    q = std::abs(q);
                    q = static_cast<float>(std::pow(q, norm));
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

inline float dot_product(const float *a, const float *b, int len) {
#if defined(__aarch64__) || defined(__arm__)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 8 <= len; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        sum0 = vfmaq_f32(sum0, va0, vb0);
        sum1 = vfmaq_f32(sum1, va1, vb1);
    }

    float32x4_t sum = vaddq_f32(sum0, sum1);
    float total = vaddvq_f32(sum); // Requires ARMv8.1-A and above

    for (; i < len; ++i) {
        total += a[i] * b[i];
    }

    return total;
#elif defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    float result[8];
    _mm256_storeu_ps(result, sum);
    float total = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];

    for (; i < len; ++i) {
        total += a[i] * b[i];
    }
    return total;
#else
#error "Unsupported architecture"
#endif
}
template <typename T>
void add_batch(const T *inp, float *Hess, float nsamples, int M, int K) { // Hess, nsamples默认是0
    const int tmp = M;
    const float ns_new = nsamples + tmp;
    const float w_old = nsamples / ns_new;
    const float w_new = 2.0f / ns_new;

    // 1. Scale existing Hessian
#pragma omp parallel for
    for (int i = 0; i < K * K; ++i) {
        Hess[i] *= w_old;
    }

    // 2. Cast input to float buffer
    std::vector<float> buffer(K * M);
#pragma omp parallel for
    for (int i = 0; i < K * M; ++i) {
        if constexpr (std::is_same<T, fp16_t>::value) {
            buffer[i] = utils::cast<float>(inp[i]);
        } else {
            buffer[i] = inp[i];
        }
    }

    // 3. Compute upper triangle of Hessian (without collapse)
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < K; ++i) {
        for (int j = i; j < K; ++j) {
            float s = dot_product(buffer.data() + i * M, buffer.data() + j * M, M);
            Hess[i * K + j] += s * w_new;
        }
    }

    // 4. Mirror to lower triangle (no collapse)
#pragma omp parallel for
    for (int i = 0; i < K; ++i) {
        for (int j = i + 1; j < K; ++j) {
            Hess[j * K + i] = Hess[i * K + j];
        }
    }
}

// Cholesky 分解 (in-place)，只支持 lower (第一步) 或 upper (第三步)

// Cholesky decomposition (lower or upper)
bool cholesky_decompose(float *A, int n, bool upper) {
    if (upper) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                float sum = A[i * n + j];
                if (j > 0) {
                    sum -= dot_product(&A[i * n], &A[j * n], j);
                }
                if (i == j) {
                    if (sum <= 0.0f) {
                        return false;
                    }
                    A[i * n + j] = std::sqrt(sum);
                } else {
                    A[i * n + j] = sum / A[j * n + j];
                }
            }
#pragma omp parallel for
            for (int j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0f;
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                float sum = A[i * n + j];
                if (j > 0) {
                    sum -= dot_product(&A[i * n], &A[j * n], j);
                }
                if (i == j) {
                    if (sum <= 0.0f) {
                        return false;
                    }
                    A[i * n + j] = std::sqrt(sum);
                } else {
                    A[i * n + j] = sum / A[j * n + j];
                }
            }
#pragma omp parallel for
            for (int j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0f;
            }
        }
    }
    return true;
}

// Compute A^{-1} from Cholesky decomposition (A = L L^T)
// A: lower-triangular Cholesky factor (n x n)
// invA: output inverse matrix (n x n), symmetric
// temp_row: temporary buffer of size n * n
void invert_symmetric_from_cholesky(const float *A, int n, float *invA, float *temp_row) {
#pragma omp parallel for
    for (int col = 0; col < n; ++col) {
        float *row_buf = temp_row + col * n;

        // Forward solve: L y = e_col
        for (int i = 0; i < n; ++i) {
            float sum = (i == col) ? 1.0f : 0.0f;
            if (i > 0) {
                sum -= dot_product(&A[i * n], row_buf, i);
            }
            row_buf[i] = sum / A[i * n + i];
        }

        // Backward solve: L^T x = y
        for (int i = n - 1; i >= 0; --i) {
            float sum = row_buf[i];
            for (int j = i + 1; j < n; ++j) {
                sum -= A[j * n + i] * invA[j * n + col];
            }
            invA[i * n + col] = sum / A[i * n + i];
        }

        // Exploit symmetry: copy upper triangle to lower
        for (int row = 0; row < col; ++row) {
            invA[col * n + row] = invA[row * n + col];
        }
    }
}

// Clear lower triangle for upper triangular result
void clear_lower_triangle(float *A, int n) {
#if defined(__aarch64__) || defined(__arm__)
    float32x4_t zero = vdupq_n_f32(0.0f);
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int j = 0;
        int row_start = i * n;
        // 每次清 4 个 float
        for (; j + 3 < i; j += 4) {
            vst1q_f32(&A[row_start + j], zero);
        }
        // 处理剩余 1~3 个
        for (; j < i; ++j) {
            A[row_start + j] = 0.0f;
        }
    }
#elif defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
    __m256 zero = _mm256_setzero_ps();
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int j = 0;
        for (; j + 7 < i; j += 8) {
            _mm256_storeu_ps(&A[i * n + j], zero);
        }
        for (; j < i; ++j) {
            A[i * n + j] = 0.0f;
        }
    }
#else
#error "Unsupported architecture"
#endif
}

void cholesky_inverse_then_upper_cholesky(float *Hess, int K) {
    cholesky_decompose(Hess, K, false);

    char *compute_workspace = (char *)malloc(2 * sizeof(float) * K * K);
    float *temp = (float *)compute_workspace; // 内存要求和32字节对齐
    float *invA = temp + K * K;
    invert_symmetric_from_cholesky(Hess, K, invA, temp);
    memcpy(Hess, invA, sizeof(float) * K * K);

    free(compute_workspace);

    cholesky_decompose(Hess, K, true);
    clear_lower_triangle(Hess, K);
}

template <typename T>
void fasterquant(T *weight, T *Q, float *Err, T *b_scale, T *zero, float *Hess,
                 int M, int K, int N,
                 int block_size = 128, float percdamp = 0.01, int group_size = -1,
                 int bits = 4, bool sym = false, bool mse = false,
                 float norm = 2.4, int grid = 100, float maxshrink = 0.8, bool sign_ed = false) {
    float maxq;
    float minq;
    if (sign_ed) { // 如果有符号量化
        maxq = static_cast<float>(std::pow(2, bits - 1) - 1);
        minq = -static_cast<float>(std::pow(2, bits - 1));
    } else {
        maxq = static_cast<float>(std::pow(2, bits) - 1);
        minq = 0.0f;
    }
    int num_groups = (group_size == -1 ? 1 : K / group_size);

    if (group_size == -1) {
        find_params(weight, b_scale, zero, N, K, bits, sym, mse, norm, grid, maxshrink, sign_ed);
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
    cholesky_inverse_then_upper_cholesky(Hess, K);

    for (int index = 0; index < K / block_size; index++) {
        for (int i = 0; i < block_size; i++) {
            float d = Hess[(index * block_size + i) * K + index * block_size + i];

            if (group_size != -1) {
                if ((index * block_size + i) % group_size == 0) {
                    int ind = (index * block_size + i) / group_size;
                    for (int n = 0; n < N; n++) {
                        find_params(&weight[n * K + index * block_size + i], &b_scale[n * num_groups + ind], &zero[n * num_groups + ind], 1, group_size, bits, sym, mse, norm, grid, maxshrink, sign_ed);
                    }
                }
            }
            int ind = (group_size != -1 ? (index * block_size + i) / group_size : 0);
            for (int n = 0; n < N; n++) {
                float q = quantize(utils::cast<float>(weight[n * K + index * block_size + i]), utils::cast<float>(b_scale[n * num_groups + ind]), utils::cast<float>(zero[n * num_groups + ind]), minq, maxq);
                if constexpr (std::is_same<T, fp16_t>::value) {
                    Q[n * K + index * block_size + i] = utils::cast<fp16_t>(q);
                } else if constexpr (std::is_same<T, float>::value) {
                    Q[n * K + index * block_size + i] = q;
                }

                float w = utils::cast<float>(weight[n * K + index * block_size + i]);
                float err = (w - q) / d;

                if (group_size == -1) {
                    for (int j = i; j < block_size; j++) {
                        if constexpr (std::is_same<T, fp16_t>::value) {
                            weight[n * K + index * block_size + j] = utils::cast<fp16_t>(utils::cast<float>(weight[n * K + index * block_size + j]) - err * Hess[(index * block_size + i) * K + j]);
                        } else if constexpr (std::is_same<T, float>::value) {
                            weight[n * K + index * block_size + j] -= err * Hess[(index * block_size + i) * K + j];
                        }
                    }
                }

                Err[n * block_size + i] = err;
            }
        }
        int i_2 = std::min((index + 1) * block_size, K);
        for (int n = 0; n < N; n++) {
            for (int j = i_2; j < K; j++) {
                float s = 0.0f;
                for (int b = 0; b < block_size; b++) {
                    s += Err[n * block_size + b] * Hess[(index * block_size + b) * K + j];
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
                         int32_t *packed_weight, int K, int N, int group_size, int bits = 4, bool sign_ed = false) {
    int maxq;
    int minq;
    if (sign_ed) { // 如果有符号量化
        maxq = int(std::pow(2, bits - 1) - 1);
        minq = -int(std::pow(2, bits - 1));
    } else {
        maxq = int(std::pow(2, bits) - 1);
        minq = 0;
    }
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
            q = std::max(minq, std::min(maxq, q)); // clamp to [minq, maxq]
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
                  int group_size, int block_size = 128) {

    float percdamp = 0.01f;

    int bits = 4;
    bool sym = false;
    bool mse = false;
    float norm = 2.4f;
    int grid = 100;
    float maxshrink = 0.8f;
    float nsamples = 0.0f;
    bool sign_ed = false;

    char *tmp = (char *)workspace + (K * K + N * block_size) * sizeof(float);
    float *Hess = (float *)workspace; //[K, K]
    float *Err = Hess + K * K;        //[N, block_size=128]
    fp16_t *Q = (fp16_t *)tmp;        //[N, K]
    fp16_t *weight = Q + N * K;       //[N, K]

    memset(Hess, 0, sizeof(float) * K * K);

    memcpy(weight, B, N * K * sizeof(fp16_t));

    add_batch<fp16_t>(A, Hess, nsamples, M, K);

    fasterquant<fp16_t>(weight, Q, Err, b_scale, zero, Hess,
                        M, K, N,
                        block_size, percdamp, group_size,
                        bits, sym, mse,
                        norm, grid, maxshrink, sign_ed);

    PackQuantizedWeight(Q, b_scale, zero, packed_weights, K, N, group_size, bits, sign_ed);
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

    int m = int(_info.m);
    int n = int(_info.n);
    int k = int(_info.k);
    int group_size = int(_info.group_size);
    int block_size = int(_info.block_size);
    bool is_weight_transposed = _info.is_weight_transposed;
    if (_info.atype == INFINI_DTYPE_F16 && !is_weight_transposed) {
        quantWeights(workspace, (int32_t *)packed_weights,
                     (fp16_t *)b_scale,
                     (fp16_t *)zero,
                     (fp16_t *)a, (fp16_t *)b, m, k, n, group_size, block_size);

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

    int m = int(_info.m);
    int n = int(_info.n);
    int k = int(_info.k);
    int group_size = int(_info.group_size);
    bool is_weight_transposed = _info.is_weight_transposed;
    if (_info.atype == INFINI_DTYPE_F16 && !is_weight_transposed) {
        caculate(workspace, (fp16_t *)c, (fp16_t *)a, (int32_t *)packed_weights, (fp16_t *)b_scale, (fp16_t *)zero,
                 m, k, n, group_size);

    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::quantize_gptq::cpu
