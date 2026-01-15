#ifndef __INFINIOP_BINARY_OP_API_H__
#define __INFINIOP_BINARY_OP_API_H__

#include "../operator_descriptor.h"

/**
 * @brief Macro to generate the C API header for a binary operator.
 * 
 * This macro generates all the necessary declarations for a binary operator:
 * - Descriptor type definition
 * - Create descriptor function
 * - Get workspace size function
 * - Execute operator function
 * - Destroy descriptor function
 * 
 * Usage:
 *   BINARY_OP_API_DECLARE(div, Div)
 *   BINARY_OP_API_DECLARE(pow, Pow)
 * 
 * @param OP_NAME      Lowercase operator name (e.g., div, pow, mod)
 * @param OP_NAME_UPPER Uppercase operator name (e.g., Div, Pow, Mod)
 */
#define BINARY_OP_API_DECLARE(OP_NAME, OP_NAME_UPPER)                        \
                                                                              \
    typedef struct InfiniopDescriptor *infiniop##OP_NAME_UPPER##Descriptor_t; \
                                                                              \
    __C __export infiniStatus_t infiniopCreate##OP_NAME_UPPER##Descriptor(  \
        infiniopHandle_t handle,                                             \
        infiniop##OP_NAME_UPPER##Descriptor_t *desc_ptr,                    \
        infiniopTensorDescriptor_t c,                                        \
        infiniopTensorDescriptor_t a,                                        \
        infiniopTensorDescriptor_t b);                                       \
                                                                              \
    __C __export infiniStatus_t infiniopGet##OP_NAME_UPPER##WorkspaceSize(  \
        infiniop##OP_NAME_UPPER##Descriptor_t desc,                         \
        size_t *size);                                                        \
                                                                              \
    __C __export infiniStatus_t infiniop##OP_NAME_UPPER(                    \
        infiniop##OP_NAME_UPPER##Descriptor_t desc,                         \
        void *workspace,                                                     \
        size_t workspace_size,                                               \
        void *c,                                                             \
        const void *a,                                                       \
        const void *b,                                                       \
        void *stream);                                                        \
                                                                              \
    __C __export infiniStatus_t infiniopDestroy##OP_NAME_UPPER##Descriptor( \
        infiniop##OP_NAME_UPPER##Descriptor_t desc);

#endif // __INFINIOP_BINARY_OP_API_H__
