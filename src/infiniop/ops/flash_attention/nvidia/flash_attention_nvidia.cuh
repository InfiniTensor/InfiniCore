#ifndef __FLASH_ATTENTION_NVIDIA_CUH__
#define __FLASH_ATTENTION_NVIDIA_CUH__

#include "../../../operator.h"
#include "../../../handle.h"
#include "../../../tensor.h"

namespace op::flash_attention::nvidia {

class Descriptor final : public InfiniopDescriptor {
public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t total_kv_len_desc,
        float scale,
        char is_causal);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *out,
        const void *q,
        const void *k,
        const void *v,
        const void *total_kv_len,
        void *stream) const;

    size_t get_workspace_size() const;

private:
    struct Opaque;
    Opaque *_opaque;

    // Tensor shape info: [batch, num_heads, seq_len, head_dim]
    std::vector<size_t> _q_shape;
    std::vector<size_t> _k_shape;
    std::vector<size_t> _v_shape;
    std::vector<size_t> _out_shape;

    // strides
    std::vector<ptrdiff_t> _q_strides;
    std::vector<ptrdiff_t> _k_strides;
    std::vector<ptrdiff_t> _v_strides;
    std::vector<ptrdiff_t> _out_strides;

    infiniDtype_t _dtype;
    float _scale;
    char _is_causal;
    size_t _batch_size;
    size_t _num_q_heads;
    size_t _num_kv_heads;
    size_t _seq_len;
    size_t _max_kv_len;
    size_t _head_dim;
    size_t _ngroup;

    Descriptor(Opaque *opaque,
               std::vector<size_t> q_shape, std::vector<ptrdiff_t> q_strides,
               std::vector<size_t> k_shape, std::vector<ptrdiff_t> k_strides,
               std::vector<size_t> v_shape, std::vector<ptrdiff_t> v_strides,
               std::vector<size_t> out_shape, std::vector<ptrdiff_t> out_strides,
               infiniDtype_t dtype, float scale, char is_causal,
               size_t batch_size, size_t num_q_heads, size_t num_kv_heads,
               size_t seq_len, size_t max_kv_len, size_t head_dim, size_t ngroup);
};

} // namespace op::flash_attention::nvidia

#endif // __FLASH_ATTENTION_NVIDIA_CUH__
