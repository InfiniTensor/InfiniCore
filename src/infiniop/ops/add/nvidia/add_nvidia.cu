#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "add_nvidia.cuh"

namespace op::add::nvidia {
namespace {

template <typename T>
INFINIOP_CUDA_KERNEL addKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    T *output,
    const T *__restrict__ a,
    const T *__restrict__ b,
    size_t offset) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= output_size) {
        return;
    }

    size_t out_idx = op::elementwise::nvidia::getOutputIndex(idx, output_contiguous, ndim, output_shape, output_strides);
    op::elementwise::nvidia::InputIndexer indexer{
        idx, ndim, input_contiguous, input_broadcasted, input_shapes, input_strides, output_strides};
    output[out_idx] = cuda::AddOp{}(a[indexer(0)], b[indexer(1)]);
}

template <typename T>
infiniStatus_t launchAddKernel(
    const op::elementwise::ElementwiseInfo &info,
    const std::shared_ptr<device::nvidia::Handle::Internal> &internal,
    void *workspace,
    void *output,
    const void *a,
    const void *b,
    cudaStream_t stream) {

    auto output_size = info.getOutputSize();
    if (output_size == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    auto ndim = info.getNdim();
    auto *d_meta_start = reinterpret_cast<int8_t *>(workspace);
    CHECK_CUDA(cudaMemcpyAsync(d_meta_start, info.getMetaStart(), info.getMetaMemSize(), cudaMemcpyHostToDevice, stream));

    auto *d_output_shape = reinterpret_cast<const size_t *>(d_meta_start);
    auto *d_output_strides = reinterpret_cast<const ptrdiff_t *>(d_output_shape + ndim);
    auto *d_input_shapes = reinterpret_cast<const size_t *>(d_output_strides + ndim);
    auto *d_input_strides = reinterpret_cast<const ptrdiff_t *>(d_input_shapes + info.getInputSize() * ndim);
    auto *d_input_contiguous = reinterpret_cast<const bool *>(d_input_strides + info.getInputSize() * ndim);
    auto *d_input_broadcasted = reinterpret_cast<const bool *>(d_input_contiguous + info.getInputSize());

    dim3 block_dims(std::min(256U, static_cast<uint32_t>(internal->maxThreadsPerBlock())));
    dim3 grid_dims(std::min(uint32_t(CEIL_DIV(output_size, block_dims.x)), static_cast<uint32_t>(internal->gridSizeX())));
    size_t step = grid_dims.x * block_dims.x;

    for (size_t i = 0; i < output_size; i += step) {
        addKernel<T><<<grid_dims, block_dims, 0, stream>>>(
            output_size,
            ndim,
            info.isOutputContiguous(),
            d_input_contiguous,
            d_input_broadcasted,
            d_output_shape,
            d_input_shapes,
            d_output_strides,
            d_input_strides,
            reinterpret_cast<T *>(output),
            reinterpret_cast<const T *>(a),
            reinterpret_cast<const T *>(b),
            i);
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
    CHECK_RESULT(info_result);
    auto info = info_result.take();
    auto workspace_size = info.getMetaMemSize();

    *desc_ptr = new Descriptor(
        dtype,
        std::move(info),
        handle->internal(),
        workspace_size,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *a,
    const void *b,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return launchAddKernel<half>(_info, _internal, workspace, output, a, b, reinterpret_cast<cudaStream_t>(stream));
    case INFINI_DTYPE_BF16:
        return launchAddKernel<cuda_bfloat16>(_info, _internal, workspace, output, a, b, reinterpret_cast<cudaStream_t>(stream));
    case INFINI_DTYPE_F32:
        return launchAddKernel<float>(_info, _internal, workspace, output, a, b, reinterpret_cast<cudaStream_t>(stream));
    case INFINI_DTYPE_I32:
        return launchAddKernel<int32_t>(_info, _internal, workspace, output, a, b, reinterpret_cast<cudaStream_t>(stream));
    case INFINI_DTYPE_I64:
        return launchAddKernel<int64_t>(_info, _internal, workspace, output, a, b, reinterpret_cast<cudaStream_t>(stream));
    case INFINI_DTYPE_F64:
        return launchAddKernel<double>(_info, _internal, workspace, output, a, b, reinterpret_cast<cudaStream_t>(stream));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::add::nvidia
