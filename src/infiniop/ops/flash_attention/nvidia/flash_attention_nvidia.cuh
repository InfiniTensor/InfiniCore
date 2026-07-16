#ifndef __FLASH_ATTENTION_NVIDIA_H__
#define __FLASH_ATTENTION_NVIDIA_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include <vector>

namespace op::flash_attention::nvidia {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(infiniopHandle_t handle,
               infiniopTensorDescriptor_t out_desc,
               infiniopTensorDescriptor_t q_desc,
               infiniopTensorDescriptor_t k_desc,
               infiniopTensorDescriptor_t v_desc,
               infiniopTensorDescriptor_t total_kv_len,
               float scale,
               char is_causal);

    ~Descriptor() = default;

    size_t get_workspace_size() const;

    infiniStatus_t calculate(void *workspace,
                             size_t workspace_size,
                             void *out,
                             const void *q,
                             const void *k,
                             const void *v,
                             const void *total_kv_len,
                             void *stream) const;

    static infiniStatus_t create(infiniopHandle_t handle,
                                 Descriptor **desc,
                                 infiniopTensorDescriptor_t out_desc,
                                 infiniopTensorDescriptor_t q_desc,
                                 infiniopTensorDescriptor_t k_desc,
                                 infiniopTensorDescriptor_t v_desc,
                                 infiniopTensorDescriptor_t total_kv_len,
                                 float scale,
                                 char is_causal);

private:
    std::vector<size_t> _out_shape;
    std::vector<ptrdiff_t> _out_strides;
    std::vector<size_t> _query_shape;
    std::vector<ptrdiff_t> _query_strides;
    std::vector<size_t> _key_shape;
    std::vector<ptrdiff_t> _key_strides;
    std::vector<size_t> _value_shape;
    std::vector<ptrdiff_t> _value_strides;
    std::vector<size_t> _total_kv_shape;
    std::vector<ptrdiff_t> _total_kv_strides;
    infiniDtype_t _dtype;
    infiniDtype_t _total_kv_dtype;
    float _scale;
    char _is_causal;
};

} // namespace op::flash_attention::nvidia

#endif // __FLASH_ATTENTION_NVIDIA_H__
