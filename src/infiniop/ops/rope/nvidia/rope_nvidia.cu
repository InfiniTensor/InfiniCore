#include "../../../devices/nvidia/nvidia_common.cuh"
#include "rope_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel.cuh"

template <typename Tdata, typename Tindex, typename Tangle>
INFINIOP_CUDA_KERNEL ropeThreadPerItemKernel(
    Tdata *y_,
    const Tdata *x_,
    const Tindex *__restrict__ pos_ids,
    const Tangle *__restrict__ sin_table,
    const Tangle *__restrict__ cos_table,
    size_t table_dim,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead) {
    ropeThreadPerItemBlock(
        y_, x_, pos_ids,
        sin_table, cos_table,
        table_dim,
        y_stride_seqlen, y_stride_nhead,
        x_stride_seqlen, x_stride_nhead);
}

namespace op::rope::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc);
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

template <typename Tdata, typename Tindex>
infiniStatus_t calculateRoPE(const RoPEInfo &info,
                             int block_size,
                             Tdata *y,
                             const Tdata *x,
                             const Tindex *pos_ids,
                             const Tdata *sin_table,
                             const Tdata *cos_table,
                             cudaStream_t stream) {
    auto dimx = uint32_t(info.seqlen),
         dimy = uint32_t(info.nhead);
    int nthreads = std::max(int(info.table_dim), block_size);

    ropeThreadPerItemKernel<<<dim3(dimx, dimy), nthreads, 0, stream>>>(
        y, x, pos_ids, sin_table, cos_table, info.table_dim,
        info.y_stride_seqlen, info.y_stride_nhead, info.x_stride_seqlen, info.x_stride_nhead);

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_ROPE(TDATA, TINDEX)                      \
    calculateRoPE(_info,                                   \
                  _opaque->internal->maxThreadsPerBlock(), \
                  (TDATA *)y,                              \
                  (const TDATA *)x,                        \
                  (const TINDEX *)pos_ids,                 \
                  (const TDATA *)sin_table,                \
                  (const TDATA *)cos_table,                \
                  (cudaStream_t)stream)

#define ROPE_TYPE(TDATA)                        \
    switch (_info.pos_type) {                   \
    case INFINI_DTYPE_U8:                       \
        return CALCULATE_ROPE(TDATA, uint8_t);  \
    case INFINI_DTYPE_U16:                      \
        return CALCULATE_ROPE(TDATA, uint16_t); \
    case INFINI_DTYPE_U32:                      \
        return CALCULATE_ROPE(TDATA, uint32_t); \
    case INFINI_DTYPE_U64:                      \
        return CALCULATE_ROPE(TDATA, uint64_t); \
    case INFINI_DTYPE_I8:                       \
        return CALCULATE_ROPE(TDATA, int8_t);   \
    case INFINI_DTYPE_I16:                      \
        return CALCULATE_ROPE(TDATA, int16_t);  \
    case INFINI_DTYPE_I32:                      \
        return CALCULATE_ROPE(TDATA, int32_t);  \
    case INFINI_DTYPE_I64:                      \
        return CALCULATE_ROPE(TDATA, int64_t);  \
    default:                                    \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;  \
    }

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const {

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        ROPE_TYPE(half);
    case INFINI_DTYPE_BF16:
        ROPE_TYPE(cuda_bfloat16);
    case INFINI_DTYPE_F32:
        ROPE_TYPE(float);
    case INFINI_DTYPE_F64:
        ROPE_TYPE(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

#undef ROPE_TYPE
#undef CALCULATE_ROPE

} // namespace op::rope::nvidia
