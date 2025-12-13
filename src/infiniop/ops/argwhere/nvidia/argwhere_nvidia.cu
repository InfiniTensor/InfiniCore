#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "argwhere_nvidia.cuh"
#include "infinicore.h"
#include <cstddef>
#include <cstdint>
#include <spdlog/spdlog.h>

infiniStatus_t launchKernel(const void *data, int64_t *results, size_t N,
                            size_t M, const size_t *shapes,
                            const ptrdiff_t *strides, size_t ndim,
                            infiniDtype_t dtype, size_t *count) {

  if (dtype == INFINI_DTYPE_F32) {
    parallel_block_argwhere_kernel<float><<<1, M / 2, M>>>(
        (float *)data, results, N, shapes, strides, ndim, count);
  } else {
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
  }
  return INFINI_STATUS_SUCCESS;
}

namespace op::argwhere::nvidia {
struct Descriptor::Opaque {
  std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(infiniopHandle_t handle,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t x_desc) {
  auto info = ArgwhereInfo::create(x_desc);
  CHECK_RESULT(info);
  size_t workspace_size = x_desc->ndim() * sizeof(size_t) * 2 +
                          x_desc->ndim() * sizeof(int64_t) * x_desc->numel() +
                          sizeof(size_t);
  *desc_ptr = new Descriptor(
      info.take(), workspace_size,
      new Opaque{
          reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
      handle->device, handle->device_id);
  return INFINI_STATUS_SUCCESS;
}

int nextPowerOfTwo(int x) {
  int power = 1;
  while (power < x) {
    power *= 2;
  }
  return power;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void **y, size_t *count, const void *x,
                                     void *stream) const {
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  size_t ndim = _info.strides.size();
  ptrdiff_t *strides = static_cast<ptrdiff_t *>(workspace);
  size_t *shapes = reinterpret_cast<size_t *>(strides + ndim);
  int64_t *result = reinterpret_cast<int64_t *>(shapes + ndim);
  size_t *count_cuda =
      reinterpret_cast<size_t *>(result + _info.num_elements * ndim);

  cudaMemcpyAsync(shapes, _info.shapes.data(),
                  _info.shapes.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                  cuda_stream);
  cudaMemcpyAsync(strides, _info.strides.data(),
                  _info.strides.size() * sizeof(ptrdiff_t),
                  cudaMemcpyHostToDevice, cuda_stream);
  // cudaStreamSynchronize(cuda_stream);
  size_t M = nextPowerOfTwo(_info.num_elements);
  CHECK_STATUS(launchKernel(x, result, _info.num_elements, M, shapes, strides,
                            ndim, INFINI_DTYPE_F32, count_cuda));
  // 从设备内存中读取 count_cuda 的值
  cudaMemcpyAsync(count, count_cuda, sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(cuda_stream);

  //写回结果
  *y = new int64_t[(*count) * ndim];
  // cudaStreamSynchronize(cuda_stream);

  // spdlog::debug("count_cuda:{}", *count_cuda);
  //   spdlog::debug("N:{}", N);

  cudaMemcpyAsync(*y, result, sizeof(int64_t) * (*count) * ndim,
                  cudaMemcpyDeviceToHost, cuda_stream);

  // cudaStreamSynchronize(cuda_stream);
  // for (size_t i = 0; i < (*count) * ndim; i++) {
  //   spdlog::debug("(*y)[{}]:{}", i, static_cast<size_t *>(*y)[i]);
  // }
  // cudaFreeAsync(result, cuda_stream);
  // cudaFreeAsync(count_cuda, cuda_stream);
  return INFINI_STATUS_SUCCESS;
}

} // namespace op::argwhere::nvidia