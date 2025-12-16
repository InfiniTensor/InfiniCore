#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "vdot_nvidia.cuh"

namespace op::vdot::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t out_desc,
                                  infiniopTensorDescriptor_t a_desc,
                                  infiniopTensorDescriptor_t b_desc) {

  auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
  auto in_dtype = a_desc->dtype();
  auto b_dtype = b_desc->dtype();
  auto out_dtype = out_desc->dtype();

  // Inputs must be 1D vectors with same length
  if (a_desc->ndim() != 1 || b_desc->ndim() != 1) {
    return INFINI_STATUS_BAD_TENSOR_SHAPE;
  }
  if (a_desc->numel() != b_desc->numel()) {
    return INFINI_STATUS_BAD_TENSOR_SHAPE;
  }

  // Input dtypes must match and be in supported set
  CHECK_OR_RETURN(in_dtype == b_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
  CHECK_DTYPE(in_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64,
              INFINI_DTYPE_BF16);

  // Output dtype equals input dtype
  CHECK_OR_RETURN(out_dtype == in_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

  size_t length = a_desc->numel();
  ptrdiff_t a_stride = a_desc->stride(0);
  ptrdiff_t b_stride = b_desc->stride(0);

  *desc_ptr =
      new Descriptor(in_dtype, out_dtype, length, a_stride, b_stride,
                     handle->internal(), handle->device, handle->device_id);

  return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *out, const void *a, const void *b,
                                     void *stream) const {

  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  // For FP16/BF16, use CUDA kernel instead of cuBLAS
  if (_in_dtype == INFINI_DTYPE_F16 || _in_dtype == INFINI_DTYPE_BF16) {
    switch (_in_dtype) {
    case INFINI_DTYPE_F16: {
      if (workspace_size < sizeof(float)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
      }
      float *tmp_out = reinterpret_cast<float *>(workspace);
      const __half *a_h = reinterpret_cast<const __half *>(a);
      const __half *b_h = reinterpret_cast<const __half *>(b);
      constexpr unsigned int BLOCK_SIZE = 256;
      op::vdot::cuda::vdotKernel<BLOCK_SIZE, __half, float>
          <<<1, BLOCK_SIZE, 0, cuda_stream>>>(tmp_out, a_h, b_h, _length,
                                              _a_stride, _b_stride);
      CHECK_CUDA(cudaGetLastError());
      float result_f;
      CHECK_CUDA(cudaMemcpyAsync(&result_f, tmp_out, sizeof(float),
                                 cudaMemcpyDeviceToHost, cuda_stream));
      CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
      __half h_result = __float2half(result_f);
      CHECK_CUDA(cudaMemcpyAsync(out, &h_result, sizeof(__half),
                                 cudaMemcpyHostToDevice, cuda_stream));
      break;
    }
    case INFINI_DTYPE_BF16: {
      if (workspace_size < sizeof(float)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
      }
      float *tmp_out = reinterpret_cast<float *>(workspace);
      const __nv_bfloat16 *a_bf = reinterpret_cast<const __nv_bfloat16 *>(a);
      const __nv_bfloat16 *b_bf = reinterpret_cast<const __nv_bfloat16 *>(b);
      constexpr unsigned int BLOCK_SIZE = 256;
      op::vdot::cuda::vdotKernel<BLOCK_SIZE, __nv_bfloat16, float>
          <<<1, BLOCK_SIZE, 0, cuda_stream>>>(tmp_out, a_bf, b_bf, _length,
                                              _a_stride, _b_stride);
      CHECK_CUDA(cudaGetLastError());
      float result_f;
      CHECK_CUDA(cudaMemcpyAsync(&result_f, tmp_out, sizeof(float),
                                 cudaMemcpyDeviceToHost, cuda_stream));
      CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
      __nv_bfloat16 bf_result = __float2bfloat16(result_f);
      CHECK_CUDA(cudaMemcpyAsync(out, &bf_result, sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice, cuda_stream));
      break;
    }
    default:
      return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
  }

  // Use cuBLAS for F32/F64
  CHECK_STATUS(_internal->useCublas(cuda_stream, [&](cublasHandle_t handle) {
    switch (_in_dtype) {
    case INFINI_DTYPE_F32: {
      if (_a_stride == 1 && _b_stride == 1) {
        // Contiguous case: use cublasSdot
        float result;
        CHECK_CUBLAS(cublasSdot(handle, static_cast<int>(_length),
                                reinterpret_cast<const float *>(a),
                                static_cast<int>(_a_stride),
                                reinterpret_cast<const float *>(b),
                                static_cast<int>(_b_stride), &result));
        CHECK_CUDA(cudaMemcpyAsync(out, &result, sizeof(float),
                                   cudaMemcpyHostToDevice, cuda_stream));
      } else {
        // Strided case: use cublasDotEx
        float result;
        CHECK_CUBLAS(cublasDotEx(handle, static_cast<int>(_length),
                                 reinterpret_cast<const float *>(a), CUDA_R_32F,
                                 static_cast<int>(_a_stride),
                                 reinterpret_cast<const float *>(b), CUDA_R_32F,
                                 static_cast<int>(_b_stride), &result,
                                 CUDA_R_32F, CUDA_R_32F));
        CHECK_CUDA(cudaMemcpyAsync(out, &result, sizeof(float),
                                   cudaMemcpyHostToDevice, cuda_stream));
      }
      break;
    }
    case INFINI_DTYPE_F64: {
      if (_a_stride == 1 && _b_stride == 1) {
        // Contiguous case: use cublasDdot
        double result;
        CHECK_CUBLAS(cublasDdot(handle, static_cast<int>(_length),
                                reinterpret_cast<const double *>(a),
                                static_cast<int>(_a_stride),
                                reinterpret_cast<const double *>(b),
                                static_cast<int>(_b_stride), &result));
        CHECK_CUDA(cudaMemcpyAsync(out, &result, sizeof(double),
                                   cudaMemcpyHostToDevice, cuda_stream));
      } else {
        // Strided case: use cublasDotEx
        double result;
        CHECK_CUBLAS(cublasDotEx(handle, static_cast<int>(_length),
                                 reinterpret_cast<const double *>(a),
                                 CUDA_R_64F, static_cast<int>(_a_stride),
                                 reinterpret_cast<const double *>(b),
                                 CUDA_R_64F, static_cast<int>(_b_stride),
                                 &result, CUDA_R_64F, CUDA_R_64F));
        CHECK_CUDA(cudaMemcpyAsync(out, &result, sizeof(double),
                                   cudaMemcpyHostToDevice, cuda_stream));
      }
      break;
    }
    default:
      return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
  }));

  return INFINI_STATUS_SUCCESS;
}

} // namespace op::vdot::nvidia
