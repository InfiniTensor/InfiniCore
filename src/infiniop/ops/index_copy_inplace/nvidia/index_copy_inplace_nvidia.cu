#include "../../../devices/nvidia/nvidia_common.cuh"
#include "index_copy_inplace_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
// #include "../cuda/common.cuh"//用自己的迷你版本

#include "../cuda/kernel.cuh"

// template <typename Tdata, typename Tindex, typename Tangle>
// INFINIOP_CUDA_KERNEL indexcopyinplaceThreadPerItemKernel(
//     Tdata *y_,
//     const Tdata *x_,
//     const Tindex *__restrict__ pos_ids,
//     const Tangle *__restrict__ sin_table,
//     const Tangle *__restrict__ cos_table,
//     size_t table_dim,
//     ptrdiff_t y_stride_seqlen,
//     ptrdiff_t y_stride_nhead,
//     ptrdiff_t x_stride_seqlen,
//     ptrdiff_t x_stride_nhead) {
//     ropeThreadPerItemBlock(
//         y_, x_, pos_ids,
//         sin_table, cos_table,
//         table_dim,
//         y_stride_seqlen, y_stride_nhead,
//         x_stride_seqlen, x_stride_nhead);
// }

//global相当于INFINIOP_CUDA_KERNEL？？？？
template <typename Tdata>
__global__ void indexCopyInplaceKernel(
    const Tdata *input_data,
    Tdata *output_data,
    const int64_t *__restrict__ index_data,
    int dim,
    int num_dims,
    size_t index_size,
    const size_t *output_shape,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides){
        indexCopyInplaceKernelBlock(input_data, output_data, index_data,
            dim, num_dims, index_size, output_shape, output_strides, input_strides);
    }

namespace op::index_copy_inplace::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,//create方法接受高层的、与平台无关的参数
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int dim,
    infiniopTensorDescriptor_t index_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info = IndexCopyInplaceInfo::createIndexCopyInplaceInfo(input_desc, output_desc, dim, index_desc);
    CHECK_RESULT(info);

    // Create descriptor
    *desc_ptr = new Descriptor(
        info.take(),
        0,
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateIndexCopyInplace(
                             const IndexCopyInplaceInfo &info,
                             int block_size,
                             const Tdata *input,
                             Tdata *output,
                             const int64_t *index,
                             cudaStream_t stream) {
    // auto dimx = uint32_t(info.seqlen),
    //      dimy = uint32_t(info.nhead);
    // int nthreads = std::max(int(info.table_dim), block_size);

    // ropeThreadPerItemKernel<<<dim3(dimx, dimy), nthreads, 0, stream>>>(
    //     y, x, pos_ids, sin_table, cos_table, info.table_dim,
    //     info.y_stride_seqlen, info.y_stride_nhead, info.x_stride_seqlen, info.x_stride_nhead);
    //rope对(seqlen, nhead, dhead)张量操作，这部分没办法完全照搬

    dim3 blockDim(block_size);
    dim3 gridDim(info.slice_size);
    indexCopyInplaceKernel<Tdata><<<gridDim, blockDim, 0, stream>>>(
        // static_cast<const Tdata *>(input),
        // static_cast<Tdata *>(output),
        // static_cast<const int64_t *>(index),
        input, output, index,
        info.dim,
        info.output_shape.size(),
        info.index_size,
        info.output_shape.data(),
        info.output_strides.data(),
        info.input_strides.data()
    );

    return cudaPeekAtLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}
//calculateIndexCopyInplace<TDATA>(_info,  
#define CALCULATE_INDEXCOPYINPLACE(TDATA)                      \
    calculateIndexCopyInplace(_info,                                   \
                  _opaque->internal->maxThreadsPerBlock(), \
                  (const TDATA *)input,                        \
                  (TDATA *)output,                \
                  (const int64_t *)index,                \
                  (cudaStream_t)stream)

// #define ROPE_TYPE(TDATA) 因为只有一层，所以不需要，rope是有两层


infiniStatus_t Descriptor::calculate(
    // void *workspace,//这个是临时GPU，一般用来存储中间结果，这个算子用不到
    // size_t workspace_size,
    const void *input,
    void *output,
    const void *index,
    void *stream) const {

    switch (_info.data_type) {
    // case INFINI_DTYPE_F16:
    //     return CALCULATE_INDEXCOPYINPLACE(half);
    // case INFINI_DTYPE_BF16:
    //     return CALCULATE_INDEXCOPYINPLACE(cuda_bfloat16);
    case INFINI_DTYPE_F32:
        return CALCULATE_INDEXCOPYINPLACE(float);
    // case INFINI_DTYPE_F64:
    //     return CALCULATE_INDEXCOPYINPLACE(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    //return INFINI_STATUS_SUCCESS;
}

//#undef ROPE_TYPE没有这部分
#undef CALCULATE_INDEXCOPYINPLACE

} // namespace op::index_copy_inplace::nvidia
