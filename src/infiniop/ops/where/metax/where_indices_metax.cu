#include "../../../devices/metax/metax_handle.h"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/where_indices_kernel.cuh"
#include "where_indices_metax.h"
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

namespace op::where::metax {

// 封装 CUB InclusiveSum
template <class T>
static hcError_t inclusiveSum(void *workspace_ptr, size_t &workspace_len,
                              T *data, int n, hcStream_t stream) {
    return cub::DeviceScan::InclusiveSum(workspace_ptr, workspace_len, data, data,
                                         n, stream);
}

// 地址对齐到 256
static constexpr size_t align256(size_t size) { return (size + 255) & (~255); }

size_t IndicesDescriptor::workspaceSize() const {
    const auto n = static_cast<int>(_numel);

    // flags 数组大小
    size_t flags_size = align256(sizeof(int64_t) * _numel);

    // CUB scan workspace
    size_t scan_workspace = 0;
    int64_t *dummy = nullptr;
    CHECK_CUDA(inclusiveSum<int64_t>(nullptr, scan_workspace, dummy, n, nullptr));

    return flags_size + scan_workspace;
}

infiniStatus_t IndicesDescriptor::create(infiniopHandle_t handle_,
                                         IndicesDescriptor **desc_ptr,
                                         infiniopTensorDescriptor_t cond_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);

    // 检查条件必须是 bool 类型
    if (cond_desc->dtype() != INFINI_DTYPE_BOOL) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    size_t numel = cond_desc->numel();
    int ndim = static_cast<int>(cond_desc->ndim());

    std::vector<size_t> shape(ndim);
    std::vector<ptrdiff_t> strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = cond_desc->shape()[i];
        strides[i] = cond_desc->stride(i);
    }

    *desc_ptr = new IndicesDescriptor(numel, ndim, shape.data(), strides.data(),
                                      handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t IndicesDescriptor::calculate(void *workspace,
                                            size_t workspace_size,
                                            void **outputs, const void *cond,
                                            void *stream,
                                            size_t *num_true) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const bool *cond_ptr = reinterpret_cast<const bool *>(cond);

    // 分配 workspace 中的内存
    int64_t *flags = reinterpret_cast<int64_t *>(workspace);
    size_t flags_size = align256(sizeof(int64_t) * _numel);
    void *scan_workspace = static_cast<char *>(workspace) + flags_size;
    size_t scan_workspace_size = workspace_size - flags_size;

    // 复制 shape 和 strides 到设备（用于 markTrueElements）
    size_t *d_shape;
    ptrdiff_t *d_strides;
    CHECK_CUDA(cudaMallocAsync(&d_shape, sizeof(size_t) * _ndim, cuda_stream));
    CHECK_CUDA(
        cudaMallocAsync(&d_strides, sizeof(ptrdiff_t) * _ndim, cuda_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_shape, _shape, sizeof(size_t) * _ndim,
                               cudaMemcpyHostToDevice, cuda_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_strides, _strides, sizeof(ptrdiff_t) * _ndim,
                               cudaMemcpyHostToDevice, cuda_stream));

    // 阶段1: 标记 True 元素
    constexpr int BLOCK_SIZE = 256;
    int grid_size = static_cast<int>((_numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
    op::where::cuda::markTrueElements<int64_t>
        <<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(flags, cond_ptr, d_shape,
                                                    d_strides, _numel, _ndim);
    CHECK_CUDA(cudaGetLastError());

    // 阶段2: 计算前缀和（inclusive scan）
    size_t temp_workspace_size = scan_workspace_size;
    CHECK_CUDA(inclusiveSum<int64_t>(scan_workspace, temp_workspace_size, flags,
                                     static_cast<int>(_numel), cuda_stream));

    // 获取 True 元素的总数
    int64_t num_true_val = 0;
    CHECK_CUDA(cudaMemcpyAsync(&num_true_val, flags + _numel - 1, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, cuda_stream));
    CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
    *num_true = static_cast<size_t>(num_true_val);

    // 阶段3: 收集每个维度的索引
    int64_t **output_ptrs = new int64_t *[_ndim];
    for (int i = 0; i < _ndim; ++i) {
        output_ptrs[i] = reinterpret_cast<int64_t *>(outputs[i]);
    }

    // d_shape / d_strides 已经在阶段1中分配和复制了，这里直接使用

    // 复制 output_ptrs 到设备
    int64_t **d_output_ptrs;
    CHECK_CUDA(
        cudaMallocAsync(&d_output_ptrs, sizeof(int64_t *) * _ndim, cuda_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, output_ptrs,
                               sizeof(int64_t *) * _ndim, cudaMemcpyHostToDevice,
                               cuda_stream));

    // 启动收集索引的 kernel
    op::where::cuda::collectIndices<int64_t>
        <<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>(
            d_output_ptrs, flags, cond_ptr, d_shape, d_strides, _numel, _ndim);
    CHECK_CUDA(cudaGetLastError());

    // 清理
    CHECK_CUDA(cudaFreeAsync(d_shape, cuda_stream));
    CHECK_CUDA(cudaFreeAsync(d_strides, cuda_stream));
    CHECK_CUDA(cudaFreeAsync(d_output_ptrs, cuda_stream));
    delete[] output_ptrs;

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::where::metax
