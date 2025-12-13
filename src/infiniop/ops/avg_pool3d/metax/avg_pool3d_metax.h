#ifndef __AVG_POOL3D_METAX_H__
#define __AVG_POOL3D_METAX_H__

#include "../../../operator.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

#ifdef ENABLE_METAX_MC_API
#include <mcdnn/mcdnn.h>
#else
#include <hcdnn/hcdnn.h>
#endif

namespace op::avg_pool3d::metax {

class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    std::unique_ptr<Opaque> _opaque;
    infiniDtype_t _dtype;

    Descriptor(infiniDtype_t dtype, std::unique_ptr<Opaque> opaque,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _opaque(std::move(opaque)) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        void *kernel_size,
        void *stride,
        void *padding);

    size_t workspaceSize() const;

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::avg_pool3d::metax

#endif // __AVG_POOL3D_METAX_H__
