#ifndef __ROTG_H__
#define __ROTG_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infiniop/ops/rotg.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::rotg::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        RotgInfo _info;                                          \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            RotgInfo info,                                       \
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
            infiniopTensorDescriptor_t y_desc,                   \
            infiniopTensorDescriptor_t c_desc,                   \
            infiniopTensorDescriptor_t s_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *x,                                             \
            void *y,                                             \
            void *c,                                             \
            void *s,                                             \
            void *stream) const;                                 \
    };                                                           \
    }

class RotgInfo {
private:
    infiniDtype_t _dtype;

    explicit RotgInfo(infiniDtype_t dtype) : _dtype(dtype) {}

public:
    inline infiniDtype_t getDtype() const { return _dtype; }

    using ResultType = utils::Result<RotgInfo>;

    static ResultType createRotgInfo(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t s_desc) {

        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(c_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(s_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto dtype = x_desc->dtype();

        CHECK_OR_RETURN(y_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(c_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(s_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_OR_RETURN(x_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(c_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(s_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        RotgInfo info(dtype);
        return ResultType(std::move(info));
    }
};

#endif // __ROTG_H__
