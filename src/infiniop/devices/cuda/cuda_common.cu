#include "cuda_handle.cuh"

namespace device::cuda {

Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>(device_id)) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

Handle::Internal::Internal(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    _warp_size = prop.warpSize;
    _max_threads_per_block = prop.maxThreadsPerBlock;
    _block_size[0] = prop.maxThreadsDim[0];
    _block_size[1] = prop.maxThreadsDim[1];
    _block_size[2] = prop.maxThreadsDim[2];
    _grid_size[0] = prop.maxGridSize[0];
    _grid_size[1] = prop.maxGridSize[1];
    _grid_size[2] = prop.maxGridSize[2];
}

infiniStatus_t Handle::Internal::useCublas(cudaStream_t stream,
                                           const Fn<cublasHandle_t> &f) const {
    auto handle = blas_handles.pop();
    if (!handle) {
        CHECK_CUBLAS(cublasCreate(&(*handle)));
    }
    CHECK_CUBLAS(cublasSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    blas_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Handle::Internal::useCudnn(cudaStream_t stream,
                                          const Fn<cudnnHandle_t> &f) const {
    auto handle = dnn_handles.pop();
    if (!handle) {
        CHECK_CUDNN(cudnnCreate(&(*handle)));
    }
    CHECK_CUDNN(cudnnSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    dnn_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t
Handle::Internal::useCusparse(cudaStream_t stream,
                              const Fn<cusparseHandle_t> &f) const {
    auto handle = sparse_handles.pop();
    if (!handle) {
        CHECK_CUSPARSE(cusparseCreate(&(*handle)));
    }
    CHECK_CUSPARSE(cusparseSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    sparse_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

int Handle::Internal::warpSize() const { return _warp_size; }
int Handle::Internal::maxThreadsPerBlock() const {
    return _max_threads_per_block;
}
int Handle::Internal::blockSizeX() const { return _block_size[0]; }
int Handle::Internal::blockSizeY() const { return _block_size[1]; }
int Handle::Internal::blockSizeZ() const { return _block_size[2]; }
int Handle::Internal::gridSizeX() const { return _grid_size[0]; }
int Handle::Internal::gridSizeY() const { return _grid_size[1]; }
int Handle::Internal::gridSizeZ() const { return _grid_size[2]; }

cudnnDataType_t getCudnnDtype(infiniDtype_t dt) {
    switch (dt) {
    case INFINI_DTYPE_F16:
        return CUDNN_DATA_HALF;
    case INFINI_DTYPE_F32:
        return CUDNN_DATA_FLOAT;
    case INFINI_DTYPE_F64:
        return CUDNN_DATA_DOUBLE;
    case INFINI_DTYPE_BF16:
        return CUDNN_DATA_BFLOAT16;
    case INFINI_DTYPE_I8:
        return CUDNN_DATA_INT8;
    case INFINI_DTYPE_I32:
        return CUDNN_DATA_INT32;
    case INFINI_DTYPE_I64:
        return CUDNN_DATA_INT64;
    case INFINI_DTYPE_U8:
        return CUDNN_DATA_UINT8;
    default:
        return CUDNN_DATA_FLOAT;
    }
}

namespace nvidia {

Handle::Handle(int device_id) : cuda::Handle(INFINI_DEVICE_NVIDIA, device_id) {}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace nvidia

} // namespace device::cuda
