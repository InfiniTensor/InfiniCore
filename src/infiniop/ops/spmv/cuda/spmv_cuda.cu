#include "../../../devices/cuda/cuda_common.cuh"
#include "../../../devices/cuda/cuda_handle.cuh"
#include "../../../devices/cuda/cuda_kernel_common.cuh"
#include "../info.h"
#include "spmv_cuda.cuh"
#include <cstdint>
#include <cusparse.h>

namespace op::spmv::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr, size_t num_cols,
                                  size_t num_rows, size_t nnz,
                                  infiniDtype_t dtype) {

    auto handle = reinterpret_cast<device::cuda::nvidia::Handle *>(handle_);

    // currently only float32 supported
    if (dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = SpMVInfo::create(num_cols, num_rows, nnz);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(dtype, result.take(), new Opaque{handle->internal()},
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *y, const void *x, const void *values,
                                     const void *row_ptr,
                                     const void *col_indices,
                                     void *stream) const {

    // do basic validation
    auto validation_result = validateSpMVCSR(y, x, values, row_ptr, col_indices, _dtype);
    CHECK_OR_RETURN(validation_result == INFINI_STATUS_SUCCESS,
                    validation_result);

    // set up data types and constants
    cudaDataType cuda_dtype = CUDA_R_32F;
    const float alpha = 1.0f, beta = 0.0f;

    CHECK_STATUS(_opaque->internal->useCusparse(
        (cudaStream_t)stream, [&](cusparseHandle_t cusparse_handle) {
            // create sparse matrix descriptor
            cusparseSpMatDescr_t mat_descr;
            CHECK_CUSPARSE(cusparseCreateCsr(
                &mat_descr, _info.num_rows, _info.num_cols, _info.nnz,
                const_cast<void *>(row_ptr), const_cast<void *>(col_indices),
                const_cast<void *>(values), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));

            // create dense vector descriptors
            cusparseDnVecDescr_t vec_x, vec_y;
            CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x, _info.num_cols,
                                               const_cast<void *>(x), cuda_dtype));
            CHECK_CUSPARSE(
                cusparseCreateDnVec(&vec_y, _info.num_rows, y, cuda_dtype));

            // compute buffer size
            size_t buffer_size = 0;
            CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                mat_descr, vec_x, &beta, vec_y, cuda_dtype,
                CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));

            // allocate external buffer if needed
            void *external_buffer = nullptr;
            if (buffer_size > 0) {
                CHECK_CUDA(cudaMalloc(&external_buffer, buffer_size));
            }

            // perform the sparse matrix-vector multiplication
            auto result = cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, mat_descr, vec_x, &beta, vec_y, cuda_dtype,
                                       CUSPARSE_SPMV_ALG_DEFAULT, external_buffer);

            // clean up resources
            if (external_buffer) {
                cudaFree(external_buffer);
            }
            cusparseDestroyDnVec(vec_x);
            cusparseDestroyDnVec(vec_y);
            cusparseDestroySpMat(mat_descr);

            CHECK_CUSPARSE(result);
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::spmv::cuda
