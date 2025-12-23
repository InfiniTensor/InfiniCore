#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_handle.h"
#include "../../../devices/moore/moore_kernel_common.h"

#include "../../../reduce/cuda/reduce.cuh"
#include "../info.h"
#include "layer_norm_moore.h"

#include <cub/block/block_reduce.cuh>

namespace op::layer_norm::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

template <unsigned int BLOCK_SIZE, typename T>
INFINIOP_MOORE_KERNEL layernormOutputKernel(
    T *__restrict__ output,
    const T *__restrict__ input,
    const T *__restrict__ weight,
    const T *__restrict__ bias,
    float eps,
    int dimsize,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    const size_t *__restrict__ shape,
    ptrdiff_t weight_stride,
    ptrdiff_t bias_stride,
    int ndim,
    bool bias_exist) {
    int ind_i = 0;
    int ind_o = 0;

    int tid = (int)blockIdx.x;
    for (int j = ndim - 2; j >= 0; j--) {
        int idx = tid % (int)shape[j];
        ind_i += idx * (int)input_strides[j];
        ind_o += idx * (int)output_strides[j];
        tid = tid / (int)shape[j];
    }

    float mu_partial = op::common_cuda::reduce_op::sum<BLOCK_SIZE, T, float>(
                           input + ind_i,
                           (size_t)dimsize)
                     / (float)dimsize;
    __shared__ float mu;
    if (threadIdx.x == 0) {
        mu = mu_partial;
    }
    __syncthreads();

    float sigma2_partial = 0.0f;
    for (int id = (int)threadIdx.x; id < dimsize; id += (int)BLOCK_SIZE) {
        float v = static_cast<float>(input[ind_i + id]) - mu;
        sigma2_partial += v * v;
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sigma2_sum = BlockReduce(temp_storage).Sum(sigma2_partial);

    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        float sigma_tmp = sqrtf(sigma2_sum * __fdividef(1.0F, (float)dimsize) + eps);
        inv_std = __fdividef(1.0F, sigma_tmp);
    }
    __syncthreads();

    for (int id = (int)threadIdx.x; id < dimsize; id += (int)BLOCK_SIZE) {
        float w = static_cast<float>(weight[id * weight_stride]);
        float b = bias_exist ? static_cast<float>(bias[id * bias_stride]) : 0.0f;
        float x = static_cast<float>(input[ind_i + id]);
        float y = w * (x - mu) * inv_std + b;
        output[ind_o + id] = static_cast<T>(y);
    }
}

template <unsigned int BLOCK_SIZE, typename T>
infiniStatus_t calculate_layer_norm(
    const LayerNormInfo &info,
    T *output,
    const T *input,
    const T *weight,
    const T *bias,
    musaStream_t stream,
    void *workspace) {
    size_t ndim = info.ndim;
    char *workspace_ptr = reinterpret_cast<char *>(workspace);

    ptrdiff_t *input_strides_dev = reinterpret_cast<ptrdiff_t *>(workspace_ptr);
    ptrdiff_t *output_strides_dev = input_strides_dev + ndim;
    ptrdiff_t *input_standardization_strides_dev = output_strides_dev + ndim;
    ptrdiff_t *input_std_deviation_strides_dev = input_standardization_strides_dev + ndim;

    size_t ptrdiff_array_size = 4 * ndim * sizeof(ptrdiff_t);
    size_t *shape_dev = reinterpret_cast<size_t *>(workspace_ptr + ptrdiff_array_size);

    CHECK_MOORE(musaMemcpyAsync(input_strides_dev, info.input_strides.data(), sizeof(ptrdiff_t) * ndim, musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(output_strides_dev, info.output_strides.data(), sizeof(ptrdiff_t) * ndim, musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(input_standardization_strides_dev, info.input_standardization_strides.data(), sizeof(ptrdiff_t) * (ndim - 1), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(input_std_deviation_strides_dev, info.input_std_deviation_strides.data(), sizeof(ptrdiff_t) * (ndim - 1), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(shape_dev, info.input_shape.data(), sizeof(size_t) * ndim, musaMemcpyHostToDevice, stream));

    int dimsize = (int)info.normalized_size;
    int num_blocks = (int)info.othersize;

    layernormOutputKernel<BLOCK_SIZE, T>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output,
            input,
            weight,
            bias,
            info.eps,
            dimsize,
            output_strides_dev,
            input_strides_dev,
            shape_dev,
            info.weight_strides[0],
            info.bias_exist ? info.bias_strides[0] : 0,
            (int)info.ndim,
            info.bias_exist);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    float eps) {
    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = LayerNormInfo::createLayerNormInfo(
        output_desc,
        input_standardization_desc,
        input_std_deviation_desc,
        input_desc,
        weight_desc,
        bias_desc,
        eps);
    CHECK_RESULT(result);
    auto info = result.take();

    size_t workspace_size = output_desc->ndim() * (sizeof(ptrdiff_t) * 4 + sizeof(size_t));

    *desc_ptr = new Descriptor(
        dtype,
        std::move(info),
        workspace_size,
        new Opaque{handle->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *input_standardization,
    void *input_std_deviation,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream_) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    (void)input_standardization;
    (void)input_std_deviation;

    musaStream_t stream = (musaStream_t)stream_;

#define CALC(BLOCK_SIZE, TDATA) \
    calculate_layer_norm<BLOCK_SIZE, TDATA>(_info, (TDATA *)output, (const TDATA *)input, (const TDATA *)weight, (const TDATA *)bias, stream, workspace)

    // Some MUSA targets report maxThreadsPerBlock() == 2048, but a 2048-thread BlockReduce
    // can exceed the shared-memory limit. Clamp to 1024/512 for compatibility.
    int max_threads = _opaque->internal->maxThreadsPerBlock();
    unsigned int block_size = (max_threads >= (int)MOORE_BLOCK_SIZE_1024) ? MOORE_BLOCK_SIZE_1024 : MOORE_BLOCK_SIZE_512;

    if (block_size == MOORE_BLOCK_SIZE_1024) {
        if (_info.dtype == INFINI_DTYPE_F16) {
            return CALC(MOORE_BLOCK_SIZE_1024, half);
        } else if (_info.dtype == INFINI_DTYPE_F32) {
            return CALC(MOORE_BLOCK_SIZE_1024, float);
        } else if (_info.dtype == INFINI_DTYPE_BF16) {
            return CALC(MOORE_BLOCK_SIZE_1024, __mt_bfloat16);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (block_size == MOORE_BLOCK_SIZE_512) {
        if (_info.dtype == INFINI_DTYPE_F16) {
            return CALC(MOORE_BLOCK_SIZE_512, half);
        } else if (_info.dtype == INFINI_DTYPE_F32) {
            return CALC(MOORE_BLOCK_SIZE_512, float);
        } else if (_info.dtype == INFINI_DTYPE_BF16) {
            return CALC(MOORE_BLOCK_SIZE_512, __mt_bfloat16);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

#undef CALC
}

} // namespace op::layer_norm::moore
