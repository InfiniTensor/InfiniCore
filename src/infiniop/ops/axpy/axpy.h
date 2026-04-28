#ifndef __AXPY_H__
#define __AXPY_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/axpy.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::axpy::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        AxpyInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            AxpyInfo info,                                       \
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
            infiniopTensorDescriptor_t alpha_desc,               \
            infiniopTensorDescriptor_t x_desc,                   \
            infiniopTensorDescriptor_t y_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            const void *alpha,                                   \
            const void *x,                                       \
            void *y,                                             \
            void *stream) const;                                 \
    };                                                           \
    }

class AxpyInfo {
private:
    size_t _size;
    ptrdiff_t _incx;
    ptrdiff_t _incy;
    infiniDtype_t _dtype;

    AxpyInfo(size_t size,
             ptrdiff_t incx,
             ptrdiff_t incy,
             infiniDtype_t dtype)
        : _size(size), _incx(incx), _incy(incy), _dtype(dtype) {}

public:
    inline size_t getSize() const { return _size; }
    inline ptrdiff_t getIncx() const { return _incx; }
    inline ptrdiff_t getIncy() const { return _incy; }
    inline infiniDtype_t getDtype() const { return _dtype; }

    using ResultType = utils::Result<AxpyInfo>;

    static ResultType createAxpyInfo(
        infiniopTensorDescriptor_t alpha_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc) {

        CHECK_OR_RETURN(alpha_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto dtype = x_desc->dtype();

        CHECK_OR_RETURN(alpha_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(x_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        CHECK_OR_RETURN(alpha_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->numel() == y_desc->numel(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto size = x_desc->numel();
        auto incx = x_desc->stride(0);
        auto incy = y_desc->stride(0);

        AxpyInfo info(size, incx, incy, dtype);
        return ResultType(std::move(info));
    }
};

#endif // __AXPY_H__