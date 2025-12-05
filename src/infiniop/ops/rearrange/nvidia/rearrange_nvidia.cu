#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "rearrange_kernel.cuh"
#include "rearrange_nvidia.cuh"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdint.h>
#include <vector>

namespace op::rearrange::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto dtype = y_desc->dtype();
    auto ndim = y_desc->ndim();

    CHECK_OR_RETURN(x_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x_desc->ndim() == ndim, INFINI_STATUS_BAD_TENSOR_SHAPE);

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();
    auto y_strides = y_desc->strides();
    auto x_strides = x_desc->strides();

    CHECK_SAME_SHAPE(x_shape, y_shape);

    auto meta = utils::RearrangeMeta::create(
        y_shape.data(),
        y_strides.data(),
        x_strides.data(),
        ndim,
        infiniSizeOf(dtype));

    CHECK_RESULT(meta);

    *desc_ptr = new Descriptor(
        std::move(*meta),
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// 简化的参数准备函数
utils::Result<RearrangeParams> prepareRearrangeParams(const utils::RearrangeMeta &meta) {
    RearrangeParams params;

    // 直接从RearrangeMeta获取参数
    params.count = meta.count();
    params.unit = meta.unit();
    params.ndim = meta.ndim();
    params.unit_size = meta.unit(); // 原始unit大小

    // 拷贝步长信息
    const ptrdiff_t *idx_strides = meta.idx_strides();
    const ptrdiff_t *dst_strides = meta.dst_strides();
    const ptrdiff_t *src_strides = meta.src_strides();

    params.idx_strides.assign(idx_strides, idx_strides + params.ndim);
    params.dst_strides.assign(dst_strides, dst_strides + params.ndim);
    params.src_strides.assign(src_strides, src_strides + params.ndim);

    return utils::Result<RearrangeParams>(params);
}

// 简化的kernel启动函数
infiniStatus_t launchKernel(
    void *y,
    const void *x,
    const RearrangeParams &params,
    cudaStream_t stream) {

    // 获取kernel函数
    auto kernel_func_result = getRearrangeKernel(params);
    CHECK_RESULT(kernel_func_result);
    auto kernel_func = kernel_func_result.take();

    if (!kernel_func) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // 计算grid和block大小
    const size_t block_size = 256; // 固定block大小，可根据设备调整
    const size_t grid_size = (params.count + block_size - 1) / block_size;

    if (grid_size == 0) {
        return INFINI_STATUS_SUCCESS; // 没有数据需要处理
    }

    // 准备kernel参数 - 需要显式转换为void*
    size_t count = params.count;
    size_t unit = params.unit;
    size_t ndim = params.ndim;

    // 为步长数组创建设备内存拷贝
    ptrdiff_t *d_idx_strides = nullptr;
    ptrdiff_t *d_dst_strides = nullptr;
    ptrdiff_t *d_src_strides = nullptr;

    cudaError_t err;

    // 分配设备内存并拷贝步长数组
    if (ndim > 0) {
        err = cudaMalloc(&d_idx_strides, ndim * sizeof(ptrdiff_t));
        if (err != cudaSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }

        err = cudaMalloc(&d_dst_strides, ndim * sizeof(ptrdiff_t));
        if (err != cudaSuccess) {
            cudaFree(d_idx_strides);
            return INFINI_STATUS_INTERNAL_ERROR;
        }

        err = cudaMalloc(&d_src_strides, ndim * sizeof(ptrdiff_t));
        if (err != cudaSuccess) {
            cudaFree(d_idx_strides);
            cudaFree(d_dst_strides);
            return INFINI_STATUS_INTERNAL_ERROR;
        }

        // 拷贝数据到设备
        err = cudaMemcpyAsync(d_idx_strides, params.idx_strides.data(),
                              ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            cudaFree(d_idx_strides);
            cudaFree(d_dst_strides);
            cudaFree(d_src_strides);
            return INFINI_STATUS_INTERNAL_ERROR;
        }

        err = cudaMemcpyAsync(d_dst_strides, params.dst_strides.data(),
                              ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            cudaFree(d_idx_strides);
            cudaFree(d_dst_strides);
            cudaFree(d_src_strides);
            return INFINI_STATUS_INTERNAL_ERROR;
        }

        err = cudaMemcpyAsync(d_src_strides, params.src_strides.data(),
                              ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            cudaFree(d_idx_strides);
            cudaFree(d_dst_strides);
            cudaFree(d_src_strides);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    }

    // 准备kernel参数
    void *args[] = {
        &y,
        &x,
        &count,
        &unit,
        &ndim,
        &d_idx_strides,
        &d_dst_strides,
        &d_src_strides};

    // 启动kernel
    err = cudaLaunchKernel(
        kernel_func,
        grid_size, block_size,
        args, 0, stream);

    // 异步释放设备内存（在stream完成后）
    if (d_idx_strides) {
        cudaFreeAsync(d_idx_strides, stream);
    }
    if (d_dst_strides) {
        cudaFreeAsync(d_dst_strides, stream);
    }
    if (d_src_strides) {
        cudaFreeAsync(d_src_strides, stream);
    }

    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 特殊情况：无维度，直接拷贝
    if (_meta.ndim() == 0 || _meta.count() == 1) {
        auto err = cudaMemcpyAsync(y, x, _meta.unit(), cudaMemcpyDeviceToDevice, cuda_stream);
        if (err != cudaSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        return INFINI_STATUS_SUCCESS;
    }

    // 准备参数
    auto params_result = prepareRearrangeParams(_meta);
    CHECK_RESULT(params_result);
    auto params = params_result.take();

    // 启动kernel
    return launchKernel(y, x, params, cuda_stream);
}

} // namespace op::rearrange::nvidia
