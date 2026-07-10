#ifndef DSV4_SGLANG_COMPRESS_C4_V2_PREFILL_H
#define DSV4_SGLANG_COMPRESS_C4_V2_PREFILL_H
#include "../../operator.h"
#include "info.h"
namespace op::dsv4_sglang_compress_c4_v2_prefill {
class Descriptor : public InfiniopDescriptor {
protected:
    Info _info;
    size_t _workspace_size;

public:
    Descriptor(Info info, size_t workspace_size, infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id}, _info(info), _workspace_size(workspace_size) {}
    size_t workspaceSize() const { return _workspace_size; }
};
} // namespace op::dsv4_sglang_compress_c4_v2_prefill
#endif
