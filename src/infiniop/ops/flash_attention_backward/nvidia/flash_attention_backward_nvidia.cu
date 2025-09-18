#include "../../../devices/nvidia/nvidia_common.cuh"
#include "flash_attention_backward_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../../flash_attention/cuda/kernel.cuh"
#include "../cuda/kernel.cuh"

template <typename Tdata>
__global__ void reduce_gradients_kernel(
    const Tdata *grad_k_expanded,
    const Tdata *grad_v_expanded,
    Tdata *grad_k,
    Tdata *grad_v,
    size_t total_seq_len,
    size_t head_dim,
    size_t group) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_seq_len) {
        for (size_t j = 0; j < head_dim; ++j) {
            Tdata sum_grad_k = 0;
            Tdata sum_grad_v = 0;
            for (size_t k = 0; k < group; ++k) {
                sum_grad_k += grad_k_expanded[i * group * head_dim + k * head_dim + j];
                sum_grad_v += grad_v_expanded[i * group * head_dim + k * head_dim + j];
            }
            grad_k[i * head_dim + j] = sum_grad_k;
            grad_v[i * head_dim + j] = sum_grad_v;
        }
    }
}

template <typename Tdata>
INFINIOP_CUDA_KERNEL flashAttentionKernel(
    Tdata *__restrict__ out_,
    Tdata *__restrict__ l_,
    const Tdata *__restrict__ q_,
    const Tdata *__restrict__ k_,
    const Tdata *__restrict__ v_,
    const float *mask_,
    const size_t seq_len_q, const size_t seq_len_kv,
    const size_t head_dim, const size_t group,
    const size_t B_r, const size_t B_c, const size_t T_r, const size_t T_c,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n,
    ptrdiff_t l_stride_b, ptrdiff_t l_stride_s, ptrdiff_t l_stride_n) {

    Tdata softmax_scale = 1.0 / sqrt(head_dim);
    flashAttentionBlock<Tdata>(
        out_, l_,
        q_, k_, v_, mask_,
        seq_len_q, seq_len_kv,
        head_dim, group,
        B_r, B_c, T_r, T_c,
        softmax_scale,
        qo_stride_b, qo_stride_s, qo_stride_n,
        kv_stride_b, kv_stride_s, kv_stride_n,
        l_stride_b, l_stride_s, l_stride_n);
}

template <typename Tdata>
INFINIOP_CUDA_KERNEL flashAttentionBackwardKernel(
    Tdata *__restrict__ grad_q_,
    Tdata *__restrict__ grad_k_,
    Tdata *__restrict__ grad_v_,
    const Tdata *__restrict__ q_,
    const Tdata *__restrict__ k_,
    const Tdata *__restrict__ v_,
    const Tdata *__restrict__ out_,
    const Tdata *__restrict__ grad_out_,
    const Tdata *__restrict__ l_,
    const float *mask_,
    const size_t seq_len_q, const size_t seq_len_kv,
    const size_t head_dim, const size_t group,
    const size_t B_r, const size_t B_c, const size_t T_r, const size_t T_c,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n,
    ptrdiff_t l_stride_b, ptrdiff_t l_stride_s, ptrdiff_t l_stride_n) {
    Tdata softmax_scale = 1.0 / sqrt(head_dim);
    flashAttentionBackwardBlock<Tdata>(
        grad_q_, grad_k_, grad_v_,
        q_, k_, v_, out_, grad_out_, l_, mask_,
        seq_len_q, seq_len_kv,
        head_dim, group,
        B_r, B_c, T_r, T_c,
        softmax_scale,
        qo_stride_b, qo_stride_s, qo_stride_n,
        kv_stride_b, kv_stride_s, kv_stride_n,
        l_stride_b, l_stride_s, l_stride_n);
}

namespace op::flash_attention_backward::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internel;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
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

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info = FlashAttentionBackwardInfo::create(grad_q_desc, grad_k_desc, grad_v_desc,
                                                   q_desc, k_desc, v_desc,
                                                   grad_out_desc, mask_desc, mask_type);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchForwardKernel(
    void *out, void *l,
    const void *q, const void *k, const void *v,
    const void *mask,
    size_t batch_size,
    size_t nums_head_q, size_t nums_head_kv,
    size_t seq_len_q, size_t seq_len_kv,
    size_t head_dim, size_t group,
    size_t B_r, size_t B_c, size_t T_r, size_t T_c,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n,
    ptrdiff_t l_stride_b, ptrdiff_t l_stride_s, ptrdiff_t l_stride_n,
    infiniDtype_t dtype,
    cudaStream_t stream) {
    cudaMemset(out, 0, batch_size * nums_head_q * seq_len_q * head_dim * sizeof(dtype));
    cudaMemset(l, 0, batch_size * nums_head_q * seq_len_q * sizeof(dtype));

    // Calculate SRAM size needed per block
    const int sram_size = (2 * B_c * head_dim * sizeof(dtype)) // SRAM size for Kj, Vj
                        + (B_r * head_dim * sizeof(dtype))     // SRAM size for Qi
                        + (B_c * B_r * sizeof(dtype));         // SRAM size for S
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
    }

    dim3 grid_dim(batch_size, nums_head_q);
    dim3 block_dim(B_r);

#define LAUNCHI_FORWARD_KERNEL(Tdata)                                        \
    flashAttentionKernel<Tdata><<<grid_dim, block_dim, sram_size, stream>>>( \
        reinterpret_cast<Tdata *>(out),                                      \
        reinterpret_cast<Tdata *>(l),                                        \
        reinterpret_cast<const Tdata *>(q),                                  \
        reinterpret_cast<const Tdata *>(k),                                  \
        reinterpret_cast<const Tdata *>(v),                                  \
        reinterpret_cast<const float *>(mask),                               \
        seq_len_q, seq_len_kv,                                               \
        head_dim, group,                                                     \
        B_r, B_c, T_r, T_c,                                                  \
        qo_stride_b, qo_stride_s, qo_stride_n,                               \
        kv_stride_b, kv_stride_s, kv_stride_n,                               \
        l_stride_b, l_stride_s, l_stride_n)

    if (dtype == INFINI_DTYPE_F16) {
        LAUNCHI_FORWARD_KERNEL(half);
    } else if (dtype == INFINI_DTYPE_F32) {
        LAUNCHI_FORWARD_KERNEL(float);
    } else if (dtype == INFINI_DTYPE_BF16) {
        LAUNCHI_FORWARD_KERNEL(__nv_bfloat16);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchBackwardKernel(
    void *grad_q, void *grad_k, void *grad_v,
    const void *q, const void *k, const void *v,
    const void *out, const void *grad_out, const void *l,
    const void *mask,
    size_t batch_size,
    size_t nums_head_q, size_t nums_head_kv,
    size_t seq_len_q, size_t seq_len_kv,
    size_t head_dim, size_t group,
    size_t B_r, size_t B_c, size_t T_r, size_t T_c,
    ptrdiff_t qo_stride_b, ptrdiff_t qo_stride_s, ptrdiff_t qo_stride_n,
    ptrdiff_t kv_stride_b, ptrdiff_t kv_stride_s, ptrdiff_t kv_stride_n,
    ptrdiff_t l_stride_b, ptrdiff_t l_stride_s, ptrdiff_t l_stride_n,
    infiniDtype_t dtype,
    cudaStream_t stream) {

    // initial grad_q, grad_k, grad_v
    cudaMemset(grad_q, 0, batch_size * nums_head_q * seq_len_q * head_dim * sizeof(dtype));
    cudaMemset(grad_k, 0, batch_size * nums_head_q * seq_len_kv * head_dim * sizeof(dtype));
    cudaMemset(grad_v, 0, batch_size * nums_head_q * seq_len_kv * head_dim * sizeof(dtype));

    // calculate SRAM size needed per block
    const int sram_size = (4 * B_c * head_dim * sizeof(dtype)) // SRAM size for K_j, V_j, dK_j, dV_j
                        + (3 * B_r * head_dim * sizeof(dtype)) // SRAM size for Q_i, O_i, dO_i
                        + (2 * B_c * B_r * sizeof(dtype));     // SRAM size for S_i, dS_i

    dim3 grad_dim(batch_size, nums_head_q);
    dim3 block_dim(B_c);

#define LAUNCHI_BACKWARD_KERNEL(Tdata)                                               \
    flashAttentionBackwardKernel<Tdata><<<grad_dim, block_dim, sram_size, stream>>>( \
        reinterpret_cast<Tdata *>(grad_q),                                           \
        reinterpret_cast<Tdata *>(grad_k),                                           \
        reinterpret_cast<Tdata *>(grad_v),                                           \
        reinterpret_cast<const Tdata *>(q),                                          \
        reinterpret_cast<const Tdata *>(k),                                          \
        reinterpret_cast<const Tdata *>(v),                                          \
        reinterpret_cast<const Tdata *>(out),                                        \
        reinterpret_cast<const Tdata *>(grad_out),                                   \
        reinterpret_cast<const Tdata *>(l),                                          \
        reinterpret_cast<const float *>(mask),                                       \
        seq_len_q, seq_len_kv,                                                       \
        head_dim, group,                                                             \
        B_r, B_c, T_r, T_c,                                                          \
        qo_stride_b, qo_stride_s, qo_stride_n,                                       \
        kv_stride_b, kv_stride_s, kv_stride_n,                                       \
        l_stride_b, l_stride_s, l_stride_n)

    if (dtype == INFINI_DTYPE_F16) {
        LAUNCHI_BACKWARD_KERNEL(half);
    } else if (dtype == INFINI_DTYPE_F32) {
        LAUNCHI_BACKWARD_KERNEL(float);
    } else if (dtype == INFINI_DTYPE_BF16) {
        LAUNCHI_BACKWARD_KERNEL(__nv_bfloat16);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *grad_q, void *grad_k, void *grad_v,
    const void *q, const void *k, const void *v,
    const void *grad_out, const void *mask,
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
            void *mask_temp;
            cudaMalloc(&mask_temp, seq_len_q * seq_len_kv * sizeof(float));
            cudaMemcpy(mask_temp, _info.mask, seq_len_q * seq_len_kv * sizeof(float), cudaMemcpyHostToDevice);
            mask_input = mask_temp;
        } else {
            mask_input = mask;
        }
    }

    size_t T_r = ceil(float(seq_len_q) / B_r);
    size_t T_c = ceil(float(seq_len_kv) / B_c);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    void *out, *l;
    if (_info.dtype == INFINI_DTYPE_F16) {
        cudaMalloc(&out, batch_size * seq_len_kv * nums_head_q * head_dim * sizeof(half));
        cudaMalloc(&l, batch_size * seq_len_kv * nums_head_q * sizeof(half));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        cudaMalloc(&out, batch_size * seq_len_kv * nums_head_q * head_dim * sizeof(float));
        cudaMalloc(&l, batch_size * seq_len_kv * nums_head_q * sizeof(float));
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        cudaMalloc(&out, batch_size * seq_len_kv * nums_head_q * head_dim * sizeof(__nv_bfloat16));
        cudaMalloc(&l, batch_size * seq_len_kv * nums_head_q * sizeof(__nv_bfloat16));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_STATUS(launchForwardKernel(
        out, l, q, k, v, mask_input,
        batch_size,
        nums_head_q, nums_head_kv,
        seq_len_q, seq_len_kv,
        head_dim, group,
        B_r, B_c, T_r, T_c,
        _info.qo_stride_b, _info.qo_stride_s, _info.qo_stride_n,
        _info.kv_stride_b, _info.kv_stride_s, _info.kv_stride_n,
        _info.l_stride_b, _info.l_stride_s, _info.l_stride_n,
        _info.dtype,
        cuda_stream));

    void *grad_k_expanded, *grad_v_expanded;
    if (_info.dtype == INFINI_DTYPE_F16) {
        cudaMalloc(&grad_k_expanded, batch_size * nums_head_kv * seq_len_kv * head_dim * group * sizeof(half));
        cudaMalloc(&grad_v_expanded, batch_size * nums_head_kv * seq_len_kv * head_dim * group * sizeof(half));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        cudaMalloc(&grad_k_expanded, batch_size * nums_head_kv * seq_len_kv * head_dim * group * sizeof(float));
        cudaMalloc(&grad_v_expanded, batch_size * nums_head_kv * seq_len_kv * head_dim * group * sizeof(float));
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        cudaMalloc(&grad_k_expanded, batch_size * nums_head_kv * seq_len_kv * head_dim * group * sizeof(__nv_bfloat16));
        cudaMalloc(&grad_v_expanded, batch_size * nums_head_kv * seq_len_kv * head_dim * group * sizeof(__nv_bfloat16));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_STATUS(launchBackwardKernel(
        grad_q, grad_k_expanded, grad_v_expanded,
        q, k, v, out, grad_out, l,
        mask_input,
        batch_size,
        nums_head_q, nums_head_kv,
        seq_len_q, seq_len_kv,
        head_dim, group,
        B_r, B_c, T_r, T_c,
        _info.qo_stride_b, _info.qo_stride_s, _info.qo_stride_n,
        _info.kv_stride_b, _info.kv_stride_s, _info.kv_stride_n,
        _info.l_stride_b, _info.l_stride_s, _info.l_stride_n,
        _info.dtype,
        cuda_stream));

    size_t total_seq_len = batch_size * nums_head_kv * seq_len_kv * head_dim;
    size_t threads_per_block = 256;
    size_t blocks = (total_seq_len + threads_per_block - 1) / threads_per_block;

    if (_info.dtype == INFINI_DTYPE_F16) {
        reduce_gradients_kernel<half><<<blocks, threads_per_block>>>(
            reinterpret_cast<const half *>(grad_k_expanded),
            reinterpret_cast<const half *>(grad_v_expanded),
            reinterpret_cast<half *>(grad_k),
            reinterpret_cast<half *>(grad_v),
            total_seq_len,
            head_dim,
            group);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        reduce_gradients_kernel<float><<<blocks, threads_per_block>>>(
            reinterpret_cast<const float *>(grad_k_expanded),
            reinterpret_cast<const float *>(grad_v_expanded),
            reinterpret_cast<float *>(grad_k),
            reinterpret_cast<float *>(grad_v),
            total_seq_len,
            head_dim,
            group);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        reduce_gradients_kernel<__nv_bfloat16><<<blocks, threads_per_block>>>(
            reinterpret_cast<const __nv_bfloat16 *>(grad_k_expanded),
            reinterpret_cast<const __nv_bfloat16 *>(grad_v_expanded),
            reinterpret_cast<__nv_bfloat16 *>(grad_k),
            reinterpret_cast<__nv_bfloat16 *>(grad_v),
            total_seq_len,
            head_dim,
            group);
    }

    cudaFree(out);
    cudaFree(l);
    cudaFree(grad_k_expanded);
    cudaFree(grad_v_expanded);

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::flash_attention_backward::nvidia
