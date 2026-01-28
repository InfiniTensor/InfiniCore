#ifndef __INFINIOP_OPERATOR_IMPL_H__
#define __INFINIOP_OPERATOR_IMPL_H__

#include "handle.h"
#include "operator.h"

// Conditional compilation helpers
#ifdef ENABLE_CPU_API
#define IF_ENABLE_CPU_API(...) __VA_ARGS__
#else
#define IF_ENABLE_CPU_API(...)
#endif

#ifdef ENABLE_NVIDIA_API
#define IF_ENABLE_NVIDIA_API(...) __VA_ARGS__
#else
#define IF_ENABLE_NVIDIA_API(...)
#endif

#ifdef ENABLE_ILUVATAR_API
#define IF_ENABLE_ILUVATAR_API(...) __VA_ARGS__
#else
#define IF_ENABLE_ILUVATAR_API(...)
#endif

#ifdef ENABLE_QY_API
#define IF_ENABLE_QY_API(...) __VA_ARGS__
#else
#define IF_ENABLE_QY_API(...)
#endif

#ifdef ENABLE_METAX_API
#define IF_ENABLE_METAX_API(...) __VA_ARGS__
#else
#define IF_ENABLE_METAX_API(...)
#endif

#ifdef ENABLE_KUNLUN_API
#define IF_ENABLE_KUNLUN_API(...) __VA_ARGS__
#else
#define IF_ENABLE_KUNLUN_API(...)
#endif

#ifdef ENABLE_CAMBRICON_API
#define IF_ENABLE_CAMBRICON_API(...) __VA_ARGS__
#else
#define IF_ENABLE_CAMBRICON_API(...)
#endif

#ifdef ENABLE_MOORE_API
#define IF_ENABLE_MOORE_API(...) __VA_ARGS__
#else
#define IF_ENABLE_MOORE_API(...)
#endif

/**
 * Binary operator implementation macros
 */
#define BINARY_OP_IMPL_CASE(OP_NAME, DEVICE, NAMESPACE, c_desc, a_desc, b_desc) \
    IF_ENABLE_##DEVICE##_API(                                                   \
        case INFINI_DEVICE_##DEVICE                                             \
        : return op::OP_NAME::NAMESPACE::Descriptor::create(                    \
            handle,                                                             \
            reinterpret_cast<op::OP_NAME::NAMESPACE::Descriptor **>(desc_ptr),  \
            c_desc,                                                             \
            {a_desc, b_desc});)

#define BINARY_OP_IMPL_DEVICE_CASES(OP_NAME, c_desc, a_desc, b_desc)       \
    BINARY_OP_IMPL_CASE(OP_NAME, CPU, cpu, c_desc, a_desc, b_desc)         \
    BINARY_OP_IMPL_CASE(OP_NAME, NVIDIA, nvidia, c_desc, a_desc, b_desc)   \
    BINARY_OP_IMPL_CASE(OP_NAME, ILUVATAR, nvidia, c_desc, a_desc, b_desc) \
    BINARY_OP_IMPL_CASE(OP_NAME, QY, nvidia, c_desc, a_desc, b_desc)       \
    BINARY_OP_IMPL_CASE(OP_NAME, METAX, metax, c_desc, a_desc, b_desc)     \
    BINARY_OP_IMPL_CASE(OP_NAME, KUNLUN, kunlun, c_desc, a_desc, b_desc)   \
    BINARY_OP_IMPL_CASE(OP_NAME, CAMBRICON, bang, c_desc, a_desc, b_desc)  \
    BINARY_OP_IMPL_CASE(OP_NAME, MOORE, moore, c_desc, a_desc, b_desc)

#define BINARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, DEVICE, NAMESPACE)                              \
    IF_ENABLE_##DEVICE##_API(                                                                      \
        case INFINI_DEVICE_##DEVICE                                                                \
        :                                                                                          \
            *size = reinterpret_cast<op::OP_NAME::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;)

#define BINARY_OP_IMPL_GET_WORKSPACE_CASES(OP_NAME)              \
    BINARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, CPU, cpu)         \
    BINARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, NVIDIA, nvidia)   \
    BINARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, ILUVATAR, nvidia) \
    BINARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, QY, nvidia)       \
    BINARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, METAX, metax)     \
    BINARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, KUNLUN, kunlun)   \
    BINARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, CAMBRICON, bang)  \
    BINARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, MOORE, moore)

#define BINARY_OP_IMPL_CALCULATE_CASE(OP_NAME, DEVICE, NAMESPACE, c, a, b)          \
    IF_ENABLE_##DEVICE##_API(                                                       \
        case INFINI_DEVICE_##DEVICE                                                 \
        : return reinterpret_cast<const op::OP_NAME::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, c, {a, b}, stream);)

#define BINARY_OP_IMPL_CALCULATE_CASES(OP_NAME, c, a, b)              \
    BINARY_OP_IMPL_CALCULATE_CASE(OP_NAME, CPU, cpu, c, a, b)         \
    BINARY_OP_IMPL_CALCULATE_CASE(OP_NAME, NVIDIA, nvidia, c, a, b)   \
    BINARY_OP_IMPL_CALCULATE_CASE(OP_NAME, ILUVATAR, nvidia, c, a, b) \
    BINARY_OP_IMPL_CALCULATE_CASE(OP_NAME, QY, nvidia, c, a, b)       \
    BINARY_OP_IMPL_CALCULATE_CASE(OP_NAME, METAX, metax, c, a, b)     \
    BINARY_OP_IMPL_CALCULATE_CASE(OP_NAME, KUNLUN, kunlun, c, a, b)   \
    BINARY_OP_IMPL_CALCULATE_CASE(OP_NAME, CAMBRICON, bang, c, a, b)  \
    BINARY_OP_IMPL_CALCULATE_CASE(OP_NAME, MOORE, moore, c, a, b)

#define BINARY_OP_IMPL_DESTROY_CASE(OP_NAME, DEVICE, NAMESPACE)                      \
    IF_ENABLE_##DEVICE##_API(                                                        \
        case INFINI_DEVICE_##DEVICE                                                  \
        : delete reinterpret_cast<const op::OP_NAME::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;)

#define BINARY_OP_IMPL_DESTROY_CASES(OP_NAME)              \
    BINARY_OP_IMPL_DESTROY_CASE(OP_NAME, CPU, cpu)         \
    BINARY_OP_IMPL_DESTROY_CASE(OP_NAME, NVIDIA, nvidia)   \
    BINARY_OP_IMPL_DESTROY_CASE(OP_NAME, ILUVATAR, nvidia) \
    BINARY_OP_IMPL_DESTROY_CASE(OP_NAME, QY, nvidia)       \
    BINARY_OP_IMPL_DESTROY_CASE(OP_NAME, METAX, metax)     \
    BINARY_OP_IMPL_DESTROY_CASE(OP_NAME, KUNLUN, kunlun)   \
    BINARY_OP_IMPL_DESTROY_CASE(OP_NAME, CAMBRICON, bang)  \
    BINARY_OP_IMPL_DESTROY_CASE(OP_NAME, MOORE, moore)

#define BINARY_OP_IMPL(OP_NAME, OP_NAME_UPPER)                           \
    __C infiniStatus_t infiniopCreate##OP_NAME_UPPER##Descriptor(        \
        infiniopHandle_t handle,                                         \
        infiniop##OP_NAME_UPPER##Descriptor_t *desc_ptr,                 \
        infiniopTensorDescriptor_t c_desc,                               \
        infiniopTensorDescriptor_t a_desc,                               \
        infiniopTensorDescriptor_t b_desc) {                             \
        switch (handle->device) {                                        \
            BINARY_OP_IMPL_DEVICE_CASES(OP_NAME, c_desc, a_desc, b_desc) \
        default:                                                         \
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;              \
        }                                                                \
    }                                                                    \
    __C infiniStatus_t infiniopGet##OP_NAME_UPPER##WorkspaceSize(        \
        infiniop##OP_NAME_UPPER##Descriptor_t desc,                      \
        size_t *size) {                                                  \
        switch (desc->device_type) {                                     \
            BINARY_OP_IMPL_GET_WORKSPACE_CASES(OP_NAME)                  \
        default:                                                         \
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;              \
        }                                                                \
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;                  \
    }                                                                    \
    __C infiniStatus_t infiniop##OP_NAME_UPPER(                          \
        infiniop##OP_NAME_UPPER##Descriptor_t desc,                      \
        void *workspace,                                                 \
        size_t workspace_size,                                           \
        void *c,                                                         \
        const void *a,                                                   \
        const void *b,                                                   \
        void *stream) {                                                  \
        switch (desc->device_type) {                                     \
            BINARY_OP_IMPL_CALCULATE_CASES(OP_NAME, c, a, b)             \
        default:                                                         \
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;              \
        }                                                                \
    }                                                                    \
    __C infiniStatus_t infiniopDestroy##OP_NAME_UPPER##Descriptor(       \
        infiniop##OP_NAME_UPPER##Descriptor_t desc) {                    \
        switch (desc->device_type) {                                     \
            BINARY_OP_IMPL_DESTROY_CASES(OP_NAME)                        \
        default:                                                         \
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;              \
        }                                                                \
    }

/**
 * Unary operator implementation macros
 */
#define UNARY_OP_IMPL_CASE(OP_NAME, DEVICE, NAMESPACE, y_desc, x_desc)         \
    IF_ENABLE_##DEVICE##_API(                                                  \
        case INFINI_DEVICE_##DEVICE                                            \
        : return op::OP_NAME::NAMESPACE::Descriptor::create(                   \
            handle,                                                            \
            reinterpret_cast<op::OP_NAME::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                            \
            {x_desc});)

#define UNARY_OP_IMPL_DEVICE_CASES(OP_NAME, y_desc, x_desc)       \
    UNARY_OP_IMPL_CASE(OP_NAME, CPU, cpu, y_desc, x_desc)         \
    UNARY_OP_IMPL_CASE(OP_NAME, NVIDIA, nvidia, y_desc, x_desc)   \
    UNARY_OP_IMPL_CASE(OP_NAME, ILUVATAR, nvidia, y_desc, x_desc) \
    UNARY_OP_IMPL_CASE(OP_NAME, QY, nvidia, y_desc, x_desc)       \
    UNARY_OP_IMPL_CASE(OP_NAME, METAX, metax, y_desc, x_desc)     \
    UNARY_OP_IMPL_CASE(OP_NAME, KUNLUN, kunlun, y_desc, x_desc)   \
    UNARY_OP_IMPL_CASE(OP_NAME, CAMBRICON, bang, y_desc, x_desc)  \
    UNARY_OP_IMPL_CASE(OP_NAME, MOORE, moore, y_desc, x_desc)

#define UNARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, DEVICE, NAMESPACE)                               \
    IF_ENABLE_##DEVICE##_API(                                                                      \
        case INFINI_DEVICE_##DEVICE                                                                \
        :                                                                                          \
            *size = reinterpret_cast<op::OP_NAME::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;)

#define UNARY_OP_IMPL_GET_WORKSPACE_CASES(OP_NAME)              \
    UNARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, CPU, cpu)         \
    UNARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, NVIDIA, nvidia)   \
    UNARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, ILUVATAR, nvidia) \
    UNARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, QY, nvidia)       \
    UNARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, METAX, metax)     \
    UNARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, KUNLUN, kunlun)   \
    UNARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, CAMBRICON, bang)  \
    UNARY_OP_IMPL_GET_WORKSPACE_CASE(OP_NAME, MOORE, moore)

#define UNARY_OP_IMPL_CALCULATE_CASE(OP_NAME, DEVICE, NAMESPACE, y, x)              \
    IF_ENABLE_##DEVICE##_API(                                                       \
        case INFINI_DEVICE_##DEVICE                                                 \
        : return reinterpret_cast<const op::OP_NAME::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, {x}, stream);)

#define UNARY_OP_IMPL_CALCULATE_CASES(OP_NAME, y, x)              \
    UNARY_OP_IMPL_CALCULATE_CASE(OP_NAME, CPU, cpu, y, x)         \
    UNARY_OP_IMPL_CALCULATE_CASE(OP_NAME, NVIDIA, nvidia, y, x)   \
    UNARY_OP_IMPL_CALCULATE_CASE(OP_NAME, ILUVATAR, nvidia, y, x) \
    UNARY_OP_IMPL_CALCULATE_CASE(OP_NAME, QY, nvidia, y, x)       \
    UNARY_OP_IMPL_CALCULATE_CASE(OP_NAME, METAX, metax, y, x)     \
    UNARY_OP_IMPL_CALCULATE_CASE(OP_NAME, KUNLUN, kunlun, y, x)   \
    UNARY_OP_IMPL_CALCULATE_CASE(OP_NAME, CAMBRICON, bang, y, x)  \
    UNARY_OP_IMPL_CALCULATE_CASE(OP_NAME, MOORE, moore, y, x)

#define UNARY_OP_IMPL_DESTROY_CASE(OP_NAME, DEVICE, NAMESPACE)                       \
    IF_ENABLE_##DEVICE##_API(                                                        \
        case INFINI_DEVICE_##DEVICE                                                  \
        : delete reinterpret_cast<const op::OP_NAME::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;)

#define UNARY_OP_IMPL_DESTROY_CASES(OP_NAME)              \
    UNARY_OP_IMPL_DESTROY_CASE(OP_NAME, CPU, cpu)         \
    UNARY_OP_IMPL_DESTROY_CASE(OP_NAME, NVIDIA, nvidia)   \
    UNARY_OP_IMPL_DESTROY_CASE(OP_NAME, ILUVATAR, nvidia) \
    UNARY_OP_IMPL_DESTROY_CASE(OP_NAME, QY, nvidia)       \
    UNARY_OP_IMPL_DESTROY_CASE(OP_NAME, METAX, metax)     \
    UNARY_OP_IMPL_DESTROY_CASE(OP_NAME, KUNLUN, kunlun)   \
    UNARY_OP_IMPL_DESTROY_CASE(OP_NAME, CAMBRICON, bang)  \
    UNARY_OP_IMPL_DESTROY_CASE(OP_NAME, MOORE, moore)

#define UNARY_OP_IMPL(OP_NAME, OP_NAME_UPPER)                      \
    __C infiniStatus_t infiniopCreate##OP_NAME_UPPER##Descriptor(  \
        infiniopHandle_t handle,                                   \
        infiniop##OP_NAME_UPPER##Descriptor_t *desc_ptr,           \
        infiniopTensorDescriptor_t y_desc,                         \
        infiniopTensorDescriptor_t x_desc) {                       \
        switch (handle->device) {                                  \
            UNARY_OP_IMPL_DEVICE_CASES(OP_NAME, y_desc, x_desc)    \
        default:                                                   \
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;        \
        }                                                          \
    }                                                              \
    __C infiniStatus_t infiniopGet##OP_NAME_UPPER##WorkspaceSize(  \
        infiniop##OP_NAME_UPPER##Descriptor_t desc,                \
        size_t *size) {                                            \
        switch (desc->device_type) {                               \
            UNARY_OP_IMPL_GET_WORKSPACE_CASES(OP_NAME)             \
        default:                                                   \
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;        \
        }                                                          \
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;            \
    }                                                              \
    __C infiniStatus_t infiniop##OP_NAME_UPPER(                    \
        infiniop##OP_NAME_UPPER##Descriptor_t desc,                \
        void *workspace,                                           \
        size_t workspace_size,                                     \
        void *y,                                                   \
        const void *x,                                             \
        void *stream) {                                            \
        switch (desc->device_type) {                               \
            UNARY_OP_IMPL_CALCULATE_CASES(OP_NAME, y, x)           \
        default:                                                   \
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;        \
        }                                                          \
    }                                                              \
    __C infiniStatus_t infiniopDestroy##OP_NAME_UPPER##Descriptor( \
        infiniop##OP_NAME_UPPER##Descriptor_t desc) {              \
        switch (desc->device_type) {                               \
            UNARY_OP_IMPL_DESTROY_CASES(OP_NAME)                   \
        default:                                                   \
            return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;        \
        }                                                          \
    }

#endif // __INFINIOP_OPERATOR_IMPL_H__
