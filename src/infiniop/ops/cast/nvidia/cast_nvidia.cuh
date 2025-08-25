#ifndef __CAST_NVIDIA_H__
#define __CAST_NVIDIA_H__

#include "../../../operator.h"
#include "../../../tensor.h"
#include "../../../handle.h"
#include <vector>

namespace op::cast::nvidia {

class Descriptor final : public InfiniopDescriptor {
private:
    infiniDtype_t _input_dtype;
    infiniDtype_t _output_dtype;
    size_t _workspace_size;
    struct Opaque;
    Opaque *_opaque;

    Descriptor(infiniDtype_t input_dtype, infiniDtype_t output_dtype, size_t workspace_size);

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec);

    size_t workspaceSize() const;

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};

} // namespace op::cast::nvidia

#endif // __CAST_NVIDIA_H__