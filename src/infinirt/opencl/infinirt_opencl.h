#ifndef __INFINIRT_OPENCL_H__
#define __INFINIRT_OPENCL_H__
#include "../infinirt_impl.h"

typedef void *infinirtOpenclDevice_t;
typedef void *infinirtOpenclContext_t;
typedef void *infinirtOpenclStream_t;

__C __export infiniStatus_t infinirtGetOpenclDevice(infinirtOpenclDevice_t *cl_device);
__C __export infiniStatus_t infinirtGetOpenclContext(infinirtOpenclContext_t *cl_context);
__C __export infiniStatus_t infinirtGetOpenclStream(infinirtOpenclStream_t *cl_command_queue);

#ifdef __cplusplus
namespace infinirt::opencl {
#ifdef ENABLE_OPENCL_API
infiniStatus_t init();
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::opencl
#endif // __cplusplus

#endif // __INFINIRT_OPENCL_H__
