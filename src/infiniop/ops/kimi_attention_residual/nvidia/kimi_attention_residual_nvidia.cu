#include "kimi_attention_residual_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cub/block/block_reduce.cuh>

namespace op::kimi_attention_residual::nvidia {
namespace {

constexpr unsigned int BLOCK_SIZE = 1024;
constexpr unsigned int SOURCE_GROUP_SIZE = 128;
constexpr unsigned int PARALLEL_SOURCE_LIMIT = BLOCK_SIZE / SOURCE_GROUP_SIZE;
constexpr size_t MAX_SOURCES = 16;

struct SumPair {
    float square;
    float dot;
};

struct SumPairReduce {
    __device__ SumPair operator()(const SumPair &left, const SumPair &right) const {
        return {left.square + right.square, left.dot + right.dot};
    }
};

template <typename T>
struct alignas(16) Vector8 {
    T values[8];
};

template <typename T>
INFINIOP_CUDA_KERNEL kimiAttentionResidualKernel(
    T *__restrict__ output,
    const T *__restrict__ values,
    const T *__restrict__ norm_weight,
    const T *__restrict__ projection_weight,
    size_t num_sources,
    size_t hidden_size,
    ptrdiff_t output_token_stride,
    ptrdiff_t values_token_stride,
    ptrdiff_t values_source_stride,
    float epsilon) {
    const size_t token = blockIdx.x;
    auto *output_row = output + token * output_token_stride;
    const auto *token_values = values + token * values_token_stride;
    __shared__ float probabilities[MAX_SOURCES];

    using BlockReduce = cub::BlockReduce<SumPair, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;

    for (size_t source = 0; source < num_sources; ++source) {
        const auto *source_values = token_values + source * values_source_stride;
        SumPair local{0.0f, 0.0f};
        for (size_t i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
            const float value = static_cast<float>(source_values[i]);
            local.square += value * value;
            local.dot += value * static_cast<float>(norm_weight[i])
                       * static_cast<float>(projection_weight[i]);
        }
        const auto reduced = BlockReduce(reduce_storage).Reduce(local, SumPairReduce{});
        if (threadIdx.x == 0) {
            probabilities[source] = reduced.dot
                                  * rsqrtf(reduced.square / static_cast<float>(hidden_size)
                                           + epsilon);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float maximum = probabilities[0];
        for (size_t source = 1; source < num_sources; ++source) {
            maximum = fmaxf(maximum, probabilities[source]);
        }
        float denominator = 0.0f;
        for (size_t source = 0; source < num_sources; ++source) {
            probabilities[source] = expf(probabilities[source] - maximum);
            denominator += probabilities[source];
        }
        for (size_t source = 0; source < num_sources; ++source) {
            probabilities[source] /= denominator;
        }
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float mixed = 0.0f;
        for (size_t source = 0; source < num_sources; ++source) {
            mixed += probabilities[source]
                   * static_cast<float>(token_values[source * values_source_stride + i]);
        }
        output_row[i] = static_cast<T>(mixed);
    }
}

template <typename T>
INFINIOP_CUDA_KERNEL kimiAttentionResidualVectorKernel(
    T *__restrict__ output,
    const T *__restrict__ values,
    const T *__restrict__ norm_weight,
    const T *__restrict__ projection_weight,
    size_t num_sources,
    size_t hidden_size,
    ptrdiff_t output_token_stride,
    ptrdiff_t values_token_stride,
    ptrdiff_t values_source_stride,
    float epsilon) {
    const size_t token = blockIdx.x;
    auto *output_row = output + token * output_token_stride;
    const auto *token_values = values + token * values_token_stride;
    const size_t vector_count = hidden_size / 8;
    __shared__ float probabilities[MAX_SOURCES];

    __shared__ SumPair source_sums[PARALLEL_SOURCE_LIMIT][SOURCE_GROUP_SIZE];

    const unsigned int source = threadIdx.x / SOURCE_GROUP_SIZE;
    const unsigned int source_thread = threadIdx.x % SOURCE_GROUP_SIZE;
    SumPair local{0.0f, 0.0f};
    if (source < num_sources) {
        const auto *source_values = token_values + source * values_source_stride;
        for (size_t vector_idx = source_thread; vector_idx < vector_count;
             vector_idx += SOURCE_GROUP_SIZE) {
            const auto value = reinterpret_cast<const Vector8<T> *>(source_values)[vector_idx];
            const auto norm = reinterpret_cast<const Vector8<T> *>(norm_weight)[vector_idx];
            const auto projection = reinterpret_cast<const Vector8<T> *>(projection_weight)[vector_idx];
#pragma unroll
            for (size_t i = 0; i < 8; ++i) {
                const float scalar = static_cast<float>(value.values[i]);
                local.square += scalar * scalar;
                local.dot += scalar * static_cast<float>(norm.values[i])
                           * static_cast<float>(projection.values[i]);
            }
        }
    }
    source_sums[source][source_thread] = local;
    __syncthreads();
    for (unsigned int stride = SOURCE_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (source_thread < stride) {
            source_sums[source][source_thread] = SumPairReduce{}(
                source_sums[source][source_thread],
                source_sums[source][source_thread + stride]);
        }
        __syncthreads();
    }
    if (source < num_sources && source_thread == 0) {
        const auto reduced = source_sums[source][0];
        probabilities[source] = reduced.dot
                              * rsqrtf(reduced.square / static_cast<float>(hidden_size)
                                       + epsilon);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float maximum = probabilities[0];
        for (size_t source = 1; source < num_sources; ++source) {
            maximum = fmaxf(maximum, probabilities[source]);
        }
        float denominator = 0.0f;
        for (size_t source = 0; source < num_sources; ++source) {
            probabilities[source] = expf(probabilities[source] - maximum);
            denominator += probabilities[source];
        }
        for (size_t source = 0; source < num_sources; ++source) {
            probabilities[source] /= denominator;
        }
    }
    __syncthreads();

    for (size_t vector_idx = threadIdx.x; vector_idx < vector_count; vector_idx += BLOCK_SIZE) {
        Vector8<T> output_value;
        float mixed[8]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t source_idx = 0; source_idx < num_sources; ++source_idx) {
            const auto value = reinterpret_cast<const Vector8<T> *>(
                token_values + source_idx * values_source_stride)[vector_idx];
            const float probability = probabilities[source_idx];
#pragma unroll
            for (size_t i = 0; i < 8; ++i) {
                mixed[i] += probability * static_cast<float>(value.values[i]);
            }
        }
#pragma unroll
        for (size_t i = 0; i < 8; ++i) {
            output_value.values[i] = static_cast<T>(mixed[i]);
        }
        reinterpret_cast<Vector8<T> *>(output_row)[vector_idx] = output_value;
    }
}

template <typename T>
void launch(const KimiAttentionResidualInfo &info,
            void *output,
            const void *values,
            const void *norm_weight,
            const void *projection_weight,
            cudaStream_t stream) {
    const bool vectorized = info.hidden_size % 8 == 0
                         && info.output_strides[0] % 8 == 0
                         && info.values_strides[0] % 8 == 0
                         && info.values_strides[1] % 8 == 0;
    if (vectorized && info.num_sources <= PARALLEL_SOURCE_LIMIT) {
        kimiAttentionResidualVectorKernel<T><<<info.num_tokens, BLOCK_SIZE, 0, stream>>>(
            static_cast<T *>(output), static_cast<const T *>(values),
            static_cast<const T *>(norm_weight), static_cast<const T *>(projection_weight),
            info.num_sources, info.hidden_size, info.output_strides[0],
            info.values_strides[0], info.values_strides[1], info.epsilon);
    } else {
        kimiAttentionResidualKernel<T><<<info.num_tokens, BLOCK_SIZE, 0, stream>>>(
            static_cast<T *>(output), static_cast<const T *>(values),
            static_cast<const T *>(norm_weight), static_cast<const T *>(projection_weight),
            info.num_sources, info.hidden_size, info.output_strides[0],
            info.values_strides[0], info.values_strides[1], info.epsilon);
    }
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t projection_weight_desc,
    float epsilon) {
    auto result = KimiAttentionResidualInfo::create(
        output_desc, values_desc, norm_weight_desc, projection_weight_desc, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *values,
    const void *norm_weight,
    const void *projection_weight,
    void *stream_) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.dtype == INFINI_DTYPE_F16) {
        launch<half>(_info, output, values, norm_weight, projection_weight, stream);
        return INFINI_STATUS_SUCCESS;
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        launch<__nv_bfloat16>(_info, output, values, norm_weight, projection_weight, stream);
        return INFINI_STATUS_SUCCESS;
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::kimi_attention_residual::nvidia
