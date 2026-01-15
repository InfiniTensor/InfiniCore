#ifndef __INFINIOP_UNARY_OP_API_H__
#define __INFINIOP_UNARY_OP_API_H__

#include "../operator_descriptor.h"

/**
 * @brief Macro to generate the C API header for a unary operator.
 * 
 * This macro generates all the necessary declarations for a unary operator:
 * - Descriptor type definition
 * - Create descriptor function
 * - Get workspace size function
 * - Execute operator function
 * - Destroy descriptor function
 * 
 * Usage:
 *   UNARY_OP_API_DECLARE(abs, Abs)
 *   UNARY_OP_API_DECLARE(log, Log)
 * 
 * @param OP_NAME      Lowercase operator name (e.g., abs, log, sin)
 * @param OP_NAME_UPPER Uppercase operator name (e.g., Abs, Log, Sin)
 */
#define UNARY_OP_API_DECLARE(OP_NAME, OP_NAME_UPPER)                          \
                                                                              \
    typedef struct InfiniopDescriptor *infiniop##OP_NAME_UPPER##Descriptor_t; \
                                                                              \
    __C __export infiniStatus_t infiniopCreate##OP_NAME_UPPER##Descriptor(  \
        infiniopHandle_t handle,                                             \
        infiniop##OP_NAME_UPPER##Descriptor_t *desc_ptr,                    \
        infiniopTensorDescriptor_t y,                                         \
        infiniopTensorDescriptor_t x);                                        \
                                                                              \
    __C __export infiniStatus_t infiniopGet##OP_NAME_UPPER##WorkspaceSize(  \
        infiniop##OP_NAME_UPPER##Descriptor_t desc,                         \
        size_t *size);                                                        \
                                                                              \
    __C __export infiniStatus_t infiniop##OP_NAME_UPPER(                    \
        infiniop##OP_NAME_UPPER##Descriptor_t desc,                         \
        void *workspace,                                                     \
        size_t workspace_size,                                               \
        void *y,                                                             \
        const void *x,                                                       \
        void *stream);                                                        \
                                                                              \
    __C __export infiniStatus_t infiniopDestroy##OP_NAME_UPPER##Descriptor( \
        infiniop##OP_NAME_UPPER##Descriptor_t desc);

#endif // __INFINIOP_UNARY_OP_API_H__
