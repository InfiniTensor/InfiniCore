#ifndef DSV4_SILU_MUL_MASKED_QUANT_H
#define DSV4_SILU_MUL_MASKED_QUANT_H

#include "../../operator.h"
#include "info.h"

namespace op::dsv4_silu_mul_masked_quant {

class Descriptor : public InfiniopDescriptor {
protected:
    Info _info;
    size_t _workspace_size;

public:
    Descriptor(Info info, size_t workspace_size, infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id}, _info(info), _workspace_size(workspace_size) {}

    size_t workspaceSize() const { return _workspace_size; }
};

} // namespace op::dsv4_silu_mul_masked_quant

#endif
