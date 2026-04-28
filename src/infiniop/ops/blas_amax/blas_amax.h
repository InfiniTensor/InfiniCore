#ifndef __BLAS_AMAX_H__
#define __BLAS_AMAX_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/blas_amax.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::blas_amax::NAMESPACE {                         \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        BlasAmaxInfo _info;                                      \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            BlasAmaxInfo info,                                   \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(std::move(info)),                            \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t result_desc);             \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *x,                                       \
            void *result,                                        \
            void *stream) const;                                 \
    };                                                           \
    }

class BlasAmaxInfo {
private:
    size_t _size;
    ptrdiff_t _incx;
    infiniDtype_t _dtype;

    BlasAmaxInfo(size_t size,
                 ptrdiff_t incx,
                 infiniDtype_t dtype)
        : _size(size), _incx(incx), _dtype(dtype) {}

public:
    inline size_t getSize() const { return _size; }
    inline ptrdiff_t getIncx() const { return _incx; }
    inline infiniDtype_t getDtype() const { return _dtype; }

    using ResultType = utils::Result<BlasAmaxInfo>;

    static utils::Result<BlasAmaxInfo> createBlasAmaxInfo(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t result_desc) {

        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(result_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto dtype = x_desc->dtype();
        auto itype = result_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_DTYPE(itype, INFINI_DTYPE_I32);

        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(result_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto size = x_desc->numel();
        auto incx = x_desc->stride(0);

        BlasAmaxInfo info(size, incx, dtype);
        return ResultType(std::move(info));
    }
};

#endif // __BLAS_AMAX_H__