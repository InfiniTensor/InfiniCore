#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "topk_nvidia.cuh"

#include <climits>
#include <cub/block/block_radix_sort.cuh>
#include <cub/cub.cuh>

namespace op::topk::nvidia {
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    bool batched_argmax;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t values_output_desc,
    infiniopTensorDescriptor_t indices_output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t k,
    size_t dim,
    bool largest,
    bool sorted) {
    auto result = TopKInfo::create(values_output_desc, indices_output_desc, input_desc, k, dim, largest, sorted);
    CHECK_RESULT(result);
    auto info = result.take();
    size_t workspace_size = 0;
    // Greedy sampling needs one maximum per row, not radix sorting the full
    // vocabulary. Keep this path narrow and leave other top-k contracts intact.
    const bool batched_argmax = handle->device == INFINI_DEVICE_ILUVATAR
                             && k == 1 && largest && info.ndim == 2 && dim == 1
                             && info.n_iteration >= 1 && info.n_iteration <= 512
                             && info.dim_elements >= 1 && info.dim_elements <= INT_MAX
                             && info.input_strides[1] == 1 && info.input_strides[0] >= ptrdiff_t(info.dim_elements)
                             && values_output_desc->shape() == std::vector<size_t>{info.n_iteration, 1}
                             && indices_output_desc->shape() == values_output_desc->shape()
                             && values_output_desc->dtype() == info.dtype && indices_output_desc->dtype() == INFINI_DTYPE_I32
                             && values_output_desc->strides()[0] == 1 && indices_output_desc->strides()[0] == 1
                             && (info.dtype == INFINI_DTYPE_BF16 || info.dtype == INFINI_DTYPE_F16 || info.dtype == INFINI_DTYPE_F32);

    workspace_size += (input_desc->ndim() + values_output_desc->ndim()) * (sizeof(size_t) + sizeof(ptrdiff_t));
    // 计算临时变量空间
    size_t dim_elements = input_desc->shape()[dim];
    size_t n_iteration = 1;
    for (size_t i = 0; i < input_desc->ndim(); i++) {
        if (i != dim) {
            n_iteration *= input_desc->shape()[i];
        }
    }
    size_t total = n_iteration * dim_elements;

    workspace_size += 3 * total * sizeof(uint32_t);
    workspace_size += 3 * total * sizeof(int32_t);
    workspace_size += n_iteration * k * (sizeof(uint32_t) + sizeof(int32_t));
    if (sorted) {
        workspace_size += n_iteration * k * (sizeof(uint32_t) + sizeof(int32_t));
    }
    workspace_size += 5 * n_iteration * sizeof(int32_t);

    if (batched_argmax) {
        workspace_size = info.n_iteration * ((info.dim_elements + 4095) / 4096) * sizeof(cub::KeyValuePair<int, float>);
    }
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal(), batched_argmax},
        info, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

using ArgMaxPair = cub::KeyValuePair<int, float>;
constexpr unsigned ARGMAX_THREADS = 256;
constexpr size_t ARGMAX_TILE = 4096;

template <typename T>
__global__ void batchedArgMaxPartials(const T *input, ptrdiff_t row_stride,
                                      size_t columns, size_t parts, ArgMaxPair *workspace) {
    const size_t row = blockIdx.y;
    const size_t start = size_t(blockIdx.x) * ARGMAX_TILE;
    ArgMaxPair best{INT_MAX, -CUDART_INF_F};
    cub::ArgMax better;
    for (size_t j = threadIdx.x; j < ARGMAX_TILE && start + j < columns; j += blockDim.x) {
        const int index = static_cast<int>(start + j);
        best = better(best, ArgMaxPair{index, static_cast<float>(input[row * row_stride + index])});
    }
    using Reduce = cub::BlockReduce<ArgMaxPair, ARGMAX_THREADS>;
    __shared__ typename Reduce::TempStorage scratch;
    auto winner = Reduce(scratch).Reduce(best, better);
    if (threadIdx.x == 0) {
        workspace[row * parts + blockIdx.x] = winner;
    }
}

template <typename T>
__global__ void batchedArgMaxFinish(const ArgMaxPair *workspace, size_t parts,
                                    T *values, int32_t *indices) {
    const size_t row = blockIdx.x;
    ArgMaxPair best{INT_MAX, -CUDART_INF_F};
    cub::ArgMax better;
    for (size_t j = threadIdx.x; j < parts; j += blockDim.x) {
        best = better(best, workspace[row * parts + j]);
    }
    using Reduce = cub::BlockReduce<ArgMaxPair, ARGMAX_THREADS>;
    __shared__ typename Reduce::TempStorage scratch;
    auto winner = Reduce(scratch).Reduce(best, better);
    if (threadIdx.x == 0) {
        values[row] = static_cast<T>(winner.value);
        indices[row] = winner.key;
    }
}

template <typename T>
infiniStatus_t launchBatchedArgMax(const TopKInfo &info, void *values, void *indices,
                                   const void *input, void *workspace, size_t workspace_size,
                                   cudaStream_t stream) {
    const size_t parts = (info.dim_elements + ARGMAX_TILE - 1) / ARGMAX_TILE;
    if (workspace_size < info.n_iteration * parts * sizeof(ArgMaxPair)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto partials = static_cast<ArgMaxPair *>(workspace);
    batchedArgMaxPartials<T><<<dim3(parts, info.n_iteration), ARGMAX_THREADS, 0, stream>>>(
        static_cast<const T *>(input), info.input_strides[0], info.dim_elements, parts, partials);
    batchedArgMaxFinish<T><<<info.n_iteration, ARGMAX_THREADS, 0, stream>>>(
        partials, parts, static_cast<T *>(values), static_cast<int32_t *>(indices));
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <size_t BLOCK_SIZE, int32_t SORT_ITEMS_PER_THREAD, typename Tdata>
infiniStatus_t launchKernel(
    const TopKInfo &info,
    Tdata *values_output, int32_t *indices_output, const Tdata *input,
    size_t k, size_t dim, bool largest, bool sorted,
    cudaStream_t stream, void *workspace, size_t workspace_size) {
    if (dim >= info.ndim) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (k == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    if (k > info.dim_elements) {
        return INFINI_STATUS_BAD_PARAM;
    }
    size_t input_ndim = info.ndim;
    size_t output_ndim = input_ndim;
    size_t n_iteration = info.n_iteration;
    size_t dim_elements = info.dim_elements;
    unsigned char *workspace_ptr = reinterpret_cast<unsigned char *>(workspace);
    size_t workspace_offset = 0;
    size_t *input_shape_cuda = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
    size_t *output_shape_cuda = input_shape_cuda + input_ndim;
    workspace_offset += (input_ndim + output_ndim) * sizeof(size_t);

    ptrdiff_t *input_strides_cuda = reinterpret_cast<ptrdiff_t *>(workspace_ptr + workspace_offset);
    ptrdiff_t *output_strides_cuda = input_strides_cuda + input_ndim;
    workspace_offset += (input_ndim + output_ndim) * sizeof(ptrdiff_t);

    CHECK_CUDA(cudaMemcpyAsync(input_shape_cuda, info.input_shape.data(), input_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_shape_cuda, info.output_shape.data(), output_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(input_strides_cuda, info.input_strides.data(), input_ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_strides_cuda, info.output_strides.data(), output_ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, stream));

    const int32_t total = n_iteration * dim_elements;

    uint32_t *cur_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(uint32_t);
    uint32_t *ones_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(uint32_t);
    uint32_t *zeros_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(uint32_t);

    int32_t *cur_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(int32_t);
    int32_t *ones_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(int32_t);
    int32_t *zeros_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(int32_t);

    uint32_t *sel_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * k * sizeof(uint32_t);
    int32_t *sel_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * k * sizeof(int32_t);
    uint32_t *sel_sorted_vals = nullptr;
    int32_t *sel_sorted_idx = nullptr;
    if (sorted) {
        sel_sorted_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
        workspace_offset += n_iteration * k * sizeof(uint32_t);
        sel_sorted_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
        workspace_offset += n_iteration * k * sizeof(int32_t);
    }

    int32_t *cur_n = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    int32_t *rem_k = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    int32_t *out_pos = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    int32_t *ones_count = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    int32_t *zeros_count = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    // init
    {
        size_t threads = 256;
        size_t blocks = (n_iteration + threads - 1) / threads;
        op::topk::cuda::init_row_state<<<blocks, threads, 0, stream>>>(cur_n, rem_k, out_pos, n_iteration, dim_elements, k);
    }
    // gather input -> cur
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((dim_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, n_iteration);
        op::topk::cuda::gather_rowwise<Tdata><<<grid, block, 0, stream>>>(
            input, cur_vals, cur_idx,
            n_iteration, dim_elements,
            input_ndim, dim,
            input_shape_cuda, input_strides_cuda);
    }
    // radix select/filter
    for (int bit = 31; bit >= 0; --bit) {
        {
            size_t threads = 256;
            size_t blocks = (n_iteration + threads - 1) / threads;
            op::topk::cuda::zero_row_counters<<<blocks, threads, 0, stream>>>(ones_count, zeros_count, n_iteration);
        }

        {
            dim3 block(BLOCK_SIZE);
            dim3 grid((dim_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, n_iteration);
            op::topk::cuda::partition_rowwise<BLOCK_SIZE><<<grid, block, 0, stream>>>(
                cur_vals, cur_idx,
                ones_vals, ones_idx,
                zeros_vals, zeros_idx,
                cur_n, n_iteration, dim_elements,
                bit, largest,
                ones_count, zeros_count);
        }

        {
            op::topk::cuda::decide_and_compact<BLOCK_SIZE><<<n_iteration, BLOCK_SIZE, 0, stream>>>(
                cur_vals, cur_idx,
                ones_vals, ones_idx,
                zeros_vals, zeros_idx,
                ones_count, zeros_count,
                cur_n, rem_k, out_pos,
                sel_vals, sel_idx,
                n_iteration, dim_elements, k);
        }
    }

    // append remaining

    op::topk::cuda::take_remaining<BLOCK_SIZE><<<n_iteration, BLOCK_SIZE, 0, stream>>>(
        cur_vals, cur_idx,
        cur_n, rem_k, out_pos,
        sel_vals, sel_idx,
        n_iteration, dim_elements, k);

    // sort (CUB block radix sort)
    const int32_t *final_idx = sel_idx;

    if (sorted) {
#ifdef ENABLE_HYGON_API
        op::topk::cuda::sort_selected_rowwise<<<n_iteration, 1, 0, stream>>>(
            sel_vals, sel_idx, sel_sorted_vals, sel_sorted_idx, n_iteration, k, largest);
#else
        std::vector<int> h_offsets(n_iteration + 1);
        for (size_t i = 0; i <= n_iteration; i++) {
            h_offsets[i] = i * k;
        }
        int *d_offsets;
        CHECK_CUDA(cudaMalloc(&d_offsets, (n_iteration + 1) * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), (n_iteration + 1) * sizeof(int), cudaMemcpyHostToDevice));

        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        if (!largest) {
            cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx,
                                                     n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx,
                                                     n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
        } else {
            cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx,
                                                               n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx,
                                                               n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
        }
        CHECK_CUDA(cudaFree(d_offsets));
        CHECK_CUDA(cudaFree(d_temp_storage));
#endif
        final_idx = sel_sorted_idx;
    }

    // scatter to output (strided write)
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, n_iteration);
        op::topk::cuda::scatter_to_output<Tdata><<<grid, block, 0, stream>>>(
            input, final_idx,
            values_output, indices_output,
            n_iteration, k,
            input_ndim, dim,
            input_shape_cuda, input_strides_cuda,
            output_shape_cuda, output_strides_cuda);
    }

    CHECK_CUDA(cudaGetLastError());

    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *values_output,
    void *indices_output,
    const void *input,
    size_t k,
    size_t dim,
    bool largest,
    bool sorted,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;
    if (_opaque->batched_argmax && k == 1 && dim == 1 && largest) {
        if (_info.dtype == INFINI_DTYPE_BF16) {
            return launchBatchedArgMax<__nv_bfloat16>(_info, values_output, indices_output, input, workspace, workspace_size, stream);
        } else if (_info.dtype == INFINI_DTYPE_F16) {
            return launchBatchedArgMax<half>(_info, values_output, indices_output, input, workspace, workspace_size, stream);
        } else {
            return launchBatchedArgMax<float>(_info, values_output, indices_output, input, workspace, workspace_size, stream);
        }
    }
    constexpr int ITEMS = 4;
#define CALCULATE_TOPK(BLOCK_SIZE, Tdata)                                        \
    launchKernel<BLOCK_SIZE, ITEMS, Tdata>(                                      \
        _info,                                                                   \
        (Tdata *)values_output, (int32_t *)indices_output, (const Tdata *)input, \
        k, dim, largest, sorted,                                                 \
        stream, workspace, workspace_size)

#define CALCULATE_TOPK_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                         \
        if (_info.dtype == INFINI_DTYPE_BF16)                 \
            return CALCULATE_TOPK(BLOCK_SIZE, __nv_bfloat16); \
        else if (_info.dtype == INFINI_DTYPE_F16)             \
            return CALCULATE_TOPK(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)             \
            return CALCULATE_TOPK(BLOCK_SIZE, float);         \
        else                                                  \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;            \
    }

    if (_opaque->internal->maxThreadsPerBlock() >= 256) {
        CALCULATE_TOPK_WITH_BLOCK_SIZE(256)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::topk::nvidia
