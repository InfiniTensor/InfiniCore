#ifndef DSV4_DEEPGEMM_TF32_HC_PERNORM_GEMM_H
#define DSV4_DEEPGEMM_TF32_HC_PERNORM_GEMM_H

#include "../../operator.h"
#include "info.h"

namespace op::dsv4_deepgemm_tf32_hc_pernorm_gemm {

class Descriptor : public InfiniopDescriptor {
protected:
    Info _info;
    size_t _workspace_size;

public:
    Descriptor(Info info, size_t workspace_size, infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id}, _info(info), _workspace_size(workspace_size) {}

    size_t workspaceSize() const { return _workspace_size; }
};

} // namespace op::dsv4_deepgemm_tf32_hc_pernorm_gemm

#endif
