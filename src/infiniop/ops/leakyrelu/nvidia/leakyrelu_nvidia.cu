#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"
#include "../leakyrelu.h"
#include "leakyrelu_nvidia.cuh"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

namespace op::leakyrelu::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

template <typename T>
struct MapCudaType {
    using Type = T;
};
template <>
struct MapCudaType<fp16_t> {
    using Type = half;
};
#if defined(__CUDA_BF16_TYPES_EXIST__) || defined(__CUDA_ARCH__)
template <>
struct MapCudaType<bf16_t> {
    using Type = __nv_bfloat16;
};
#endif

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    float negative_slope) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info_r = LeakyReLUInfo::create(out_desc, in_desc, negative_slope);
    CHECK_RESULT(info_r);
    auto info = info_r.take();

    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        info,
        workspace_size,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::workspaceSize() const {
    return _min_workspace_size;
}

template <typename T>
static inline infiniStatus_t cuda_leakyrelu_impl_incremental(
    void *output_, const void *input_,
    const op::leakyrelu::LeakyReLUInfo &info,
    void *stream_) {

    int bs = 256, grid = 0;
    cudaError_t propErr;
    int device_id_local = 0;
    using DevT = typename MapCudaType<T>::Type;

    auto out_dev = reinterpret_cast<DevT *>(output_);
    auto in_dev = reinterpret_cast<const DevT *>(input_);
    auto stream = reinterpret_cast<cudaStream_t>(stream_);

    int ndim = static_cast<int>(info.shape.size());
    if (ndim == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    std::vector<size_t> h_shape(info.shape.begin(), info.shape.end());
    std::vector<size_t> h_div(ndim);
    h_div[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        h_div[d] = h_div[d + 1] * h_shape[d + 1];
    }

    std::vector<long long> h_in_stride(ndim), h_out_stride(ndim);
    for (int d = 0; d < ndim; ++d) {
        h_in_stride[d] = static_cast<long long>(info.in_stride[d]);
        h_out_stride[d] = static_cast<long long>(info.out_stride[d]);
    }

    size_t *d_shape = nullptr;
    size_t *d_div = nullptr;
    long long *d_in_stride = nullptr;
    long long *d_out_stride = nullptr;

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(reinterpret_cast<void **>(&d_shape), sizeof(size_t) * ndim);
    if (err != cudaSuccess) {
        goto cleanup;
    }
    err = cudaMalloc(reinterpret_cast<void **>(&d_div), sizeof(size_t) * ndim);
    if (err != cudaSuccess) {
        goto cleanup;
    }
    err = cudaMalloc(reinterpret_cast<void **>(&d_in_stride), sizeof(long long) * ndim);
    if (err != cudaSuccess) {
        goto cleanup;
    }
    err = cudaMalloc(reinterpret_cast<void **>(&d_out_stride), sizeof(long long) * ndim);
    if (err != cudaSuccess) {
        goto cleanup;
    }
    err = cudaMemcpyAsync(d_shape, h_shape.data(), sizeof(size_t) * ndim, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }
    err = cudaMemcpyAsync(d_div, h_div.data(), sizeof(size_t) * ndim, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }
    err = cudaMemcpyAsync(d_in_stride, h_in_stride.data(), sizeof(long long) * ndim, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }
    err = cudaMemcpyAsync(d_out_stride, h_out_stride.data(), sizeof(long long) * ndim, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }

    device_id_local = 0;
    propErr = cudaGetDevice(&device_id_local);
    if (propErr == cudaSuccess) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id_local) == cudaSuccess) {
            bs = std::min(bs, static_cast<int>(prop.maxThreadsPerBlock) / 2);
        } else {
            if (bs > 256) {
                bs = 256;
            }
        }
    } else {
        if (bs > 256) {
            bs = 256;
        }
    }

    if (bs <= 0) {
        bs = 256;
    }
    grid = static_cast<int>((info.n + bs - 1) / bs);
    if (grid <= 0) {
        grid = 1;
    }

    leakyrelu_kernel<DevT><<<grid, bs, 0, stream>>>(
        out_dev, in_dev, info.n, info.negative_slope, d_shape, d_div, d_in_stride, d_out_stride, ndim);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        goto cleanup;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }

    cudaFree(d_shape);
    cudaFree(d_div);
    cudaFree(d_in_stride);
    cudaFree(d_out_stride);
    return INFINI_STATUS_SUCCESS;

cleanup:
    cudaFree(d_shape);
    cudaFree(d_div);
    cudaFree(d_in_stride);
    cudaFree(d_out_stride);
    return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    switch (_info.dt_in) {
    case INFINI_DTYPE_F16:
        cuda_leakyrelu_impl_incremental<fp16_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        cuda_leakyrelu_impl_incremental<cuda_bfloat16>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        cuda_leakyrelu_impl_incremental<float>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        cuda_leakyrelu_impl_incremental<double>(output, input, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

}; // namespace op::leakyrelu::nvidia
