#ifndef __CAST_CPU_H__
#define __CAST_CPU_H__

#include "../../../operator.h"
#include "../../../tensor.h"
#include "../../../handle.h"
#include <vector>

namespace op::cast::cpu {

class Descriptor final : public InfiniopDescriptor {
private:
    infiniDtype_t _input_dtype;
    infiniDtype_t _output_dtype;
    struct Opaque;
    Opaque *_opaque;

    Descriptor(infiniDtype_t input_dtype, infiniDtype_t output_dtype);

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

// 模板辅助函数声明
template<typename InputType, typename OutputType>
void cast_elements(const InputType* input, OutputType* output, size_t numel);

} // namespace op::cast::cpu

#endif // __CAST_CPU_H__