#include "../../../devices/nvidia/nvidia_common.cuh"
#include "2dmrope_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/mrope.cuh"

namespace op::mrope2d::nvidia {

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
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t pos_desc,
        infiniopTensorDescriptor_t sin_desc,
        infiniopTensorDescriptor_t cos_desc) {

        auto handle_nvidia = reinterpret_cast<device::nvidia::Handle *>(handle);

        auto info = MRoPE2DInfo::createMRoPE2DInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc);
        CHECK_RESULT(info);

        // Create descriptor
        *desc_ptr = new Descriptor(
            info.take(),
            0,
            new Opaque{handle_nvidia->internal()},
            handle_nvidia->device,
            handle_nvidia->device_id);
        return INFINI_STATUS_SUCCESS;
    }

    template<typename Tdata, typename Tpos>
    __global__ void mrope2d_kernel(
        Tdata *__restrict__ y_,
        int const stride_token_y,
        int const stride_head_y,
        Tdata const *__restrict__ x_,
        int const stride_token_x,
        int const stride_head_x,
        Tpos const *__restrict__ pos_,
        float const *__restrict__ sin_table,
        float const *__restrict__ cos_table,
        int const dh_div_2) {

        padding<Tpos, Tdata>(
            y_, stride_token_y, stride_head_y,
            x_, stride_token_x, stride_head_x,
            pos_, sin_table, cos_table);
    }

    template<typename Tdata, typename Tpos>
    __global__ void mrope2d_kernel(
        Tdata *__restrict__ y_,
        int const stride_token_y,
        int const stride_head_y,
        Tdata const *__restrict__ x_,
        int const stride_token_x,
        int const stride_head_x,
        Tpos const *__restrict__ pos_,
        float const *__restrict__ sin_table,
        float const *__restrict__ cos_table) {

        padding<Tpos, Tdata>(
            y_, stride_token_y, stride_head_y,
            x_, stride_token_x, stride_head_x,
            pos_, sin_table, cos_table);
    }

    template <typename Tdata, typename Tpos>
    infiniStatus_t calculateMRoPE2D(const MRoPE2DInfo &info,
                                    int block_size,
                                    Tdata *y,
                                    const Tdata *x,
                                    const Tpos *pos_ids,
                                    const float *sin_table,
                                    const float *cos_table,
                                    cudaStream_t stream) {
        auto dimy = uint32_t(info.seqlen),    // grid.y = n
             dimx = uint32_t(info.nhead);     // grid.x = nh_h
        int dh_div_2 = info.dhead / 2;
        int nh_l = 1; // 每个 block 处理的 head 数量

        // 注意：Rust 中的顺序是 (grid.y, grid.x), (block.y, block.x)
        // 所以 CUDA 调用应该是 (grid.x, grid.y), (block.x, block.y)
        dim3 gridDim(dimx, dimy);    // (nh_h, n)
        dim3 blockDim(dh_div_2, nh_l); // (dh_div_2, nh_l)

        mrope2d_kernel<<<gridDim, blockDim, 0, stream>>>(
            y, info.y_stride_seqlen, info.y_stride_nhead,
            x, info.x_stride_seqlen, info.x_stride_nhead,
            pos_ids, sin_table, cos_table);

        return INFINI_STATUS_SUCCESS;
    }

#define CALCULATE_MROPE2D(TDATA, TPOS)                     \
    calculateMRoPE2D(_info,                                \
                     _opaque->internal->maxThreadsPerBlock(), \
                     (TDATA *)y,                           \
                     (const TDATA *)x,                     \
                     (const TPOS *)pos_ids,                \
                     (const float *)sin_table,             \
                     (const float *)cos_table,             \
                     (cudaStream_t)stream)

#define MROPE2D_TYPE(TDATA)                     \
    switch (_info.pos_type) {                  \
    case INFINI_DTYPE_U8:                      \
        return CALCULATE_MROPE2D(TDATA, uint8_t);  \
    case INFINI_DTYPE_U16:                     \
        return CALCULATE_MROPE2D(TDATA, uint16_t); \
    case INFINI_DTYPE_U32:                     \
        return CALCULATE_MROPE2D(TDATA, uint32_t); \
    case INFINI_DTYPE_U64:                     \
        return CALCULATE_MROPE2D(TDATA, uint64_t); \
    case INFINI_DTYPE_I8:                      \
        return CALCULATE_MROPE2D(TDATA, int8_t);   \
    case INFINI_DTYPE_I16:                     \
        return CALCULATE_MROPE2D(TDATA, int16_t);  \
    case INFINI_DTYPE_I32:                     \
        return CALCULATE_MROPE2D(TDATA, int32_t);  \
    case INFINI_DTYPE_I64:                     \
        return CALCULATE_MROPE2D(TDATA, int64_t);  \
    default:                                   \
        return INFINI_STATUS_BAD_TENSOR_DTYPE; \
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
            MROPE2D_TYPE(half);
        case INFINI_DTYPE_BF16:
            MROPE2D_TYPE(cuda_bfloat16);
        case INFINI_DTYPE_F32:
            MROPE2D_TYPE(float);
        case INFINI_DTYPE_F64:
            MROPE2D_TYPE(double);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        return INFINI_STATUS_SUCCESS;
    }

#undef MROPE2D_TYPE
#undef CALCULATE_MROPE2D

}
