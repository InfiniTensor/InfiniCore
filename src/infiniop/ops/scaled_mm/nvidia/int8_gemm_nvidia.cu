#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "int8_gemm_nvidia.cuh"
#include "int8_gemm_kernel.cuh"

namespace op::i8gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

inline int getSMVersion() {
  int device{-1};
  CHECK_CUDA(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_CUDA(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t a_scale_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scale_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = I8GemmInfo::create(out_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    // CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        result.take(), 0, dtype,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    // float beta,
    const void *bias,
    const void *a,
    const void *a_scale,
    const void *b,
    const void *b_scale,
    void *stream) const {

    // (out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream)
    auto sm_version = getSMVersion();


    if (sm_version >= 75 && sm_version < 80) {
        // TORCH_CHECK(out_dtype == torch::kHalf, "out_dtype must be Half for SM75");
        // sm75_dispatch_shape<cutlass::half_t, cutlass::arch::Sm75, cutlass::gemm::GemmShape<8, 8, 16>>(
        //     // out, mat_a, mat_b, scales_a, scales_b, bias);
        //     out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
    } else if (sm_version >= 80 && sm_version < 90) {
        // sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
        // if (sm_version == 86 || sm_version == 89) {
        // if (out_dtype == torch::kBFloat16) {
        //     sm89_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
        //         // out, mat_a, mat_b, scales_a, scales_b, bias);
        //         out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
        // } else {
        //     sm89_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
        //         // out, mat_a, mat_b, scales_a, scales_b, bias);
        //         out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
        // }
        // } else {
        // if (out_dtype == torch::kBFloat16) {
        //     sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
        //         // out, mat_a, mat_b, scales_a, scales_b, bias);
        //         out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
        // } else {
        //     sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
        //         // out, mat_a, mat_b, scales_a, scales_b, bias);
        //         out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
        // }
        // }
    } else if (sm_version == 90) {
    #if defined CUDA_VERSION && CUDA_VERSION >= 12000
        // cutlass 3.x
        if (this->_out_dtype == INFINI_DTYPE_BF16) {
        sm90_dispatch_shape<cutlass::bfloat16_t>(
            out, a, b, a_scale, b_scale, bias, 
            _info.m, _info.n, _info.k, 
            _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), 
            stream);
        } else {
        sm90_dispatch_shape<cutlass::half_t>(
            out, a, b, a_scale, b_scale, bias, 
            _info.m, _info.n, _info.k, 
            _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), 
            stream);
        }
    #else
        // // fallback to cutlass 2.x
        // if (out_dtype == torch::kBFloat16) {
        // sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
        //     // out, mat_a, mat_b, scales_a, scales_b, bias);
        //     out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
        // } else {
        // sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
        //     // out, mat_a, mat_b, scales_a, scales_b, bias);
        //     out, a, b, a_scale, b_scale, bias, _info.m, _info.n, _info.k, _info.a_matrix.ld(), _info.b_matrix.ld(), _info.out_matrix.ld(), stream);
        // }
    #endif
    } else {
        // TORCH_CHECK_NOT_IMPLEMENTED(false, "No implemented int8_scaled_mm for current compute capability.");
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }

    // CHECK_STATUS(_opaque->internal->useCublas(
    //     (cudaStream_t)stream,
    //     [&](cublasHandle_t handle) {
    //         CHECK_CUBLAS(
    //             cublasGemmStridedBatchedEx(
    //                 handle,
    //                 op_a,
    //                 op_b,
    //                 static_cast<int>(_info.m),
    //                 static_cast<int>(_info.n),
    //                 static_cast<int>(_info.k),
    //                 &alpha,
    //                 a,
    //                 a_type,
    //                 static_cast<int>(_info.a_matrix.ld()),
    //                 _info.a_matrix.stride,
    //                 b,
    //                 b_type,
    //                 static_cast<int>(_info.b_matrix.ld()),
    //                 _info.b_matrix.stride,
    //                 &beta,
    //                 c,
    //                 c_type,
    //                 static_cast<int>(_info.c_matrix.ld()),
    //                 _info.c_matrix.stride,
    //                 static_cast<int>(_info.batch),
    //                 compute_type,
    //                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    //         return INFINI_STATUS_SUCCESS;
    //     }));
    return INFINI_STATUS_SUCCESS;
}




} // namespace op::gemm::nvidia
