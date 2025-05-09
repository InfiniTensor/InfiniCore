#ifndef __MATMUL_GPTQ_CPU_H__
#define __MATMUL_GPTQ_CPU_H__

#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::matmul_gptq::cpu {

class Descriptor : public InfiniopDescriptor {
    int _m, _n, _k;
    size_t _workspace_size;
    infiniDtype_t _atype;
    int _num_groups;
    int _group_size;

    Descriptor(int m, int n, int k,
               size_t workspace_size,
               infiniDtype_t atype, int num_groups, int group_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _m(m), _n(n), _k(k), _workspace_size(workspace_size),
          _atype(atype), _num_groups(num_groups), _group_size(group_size) {}

public:
    ~Descriptor();
    size_t minWorkspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                 infiniopTensorDescriptor_t c_desc,
                                 infiniopTensorDescriptor_t a_desc,
                                 infiniopTensorDescriptor_t packed_weights_desc,
                                 infiniopTensorDescriptor_t b_scale_desc,
                                 infiniopTensorDescriptor_t zero_desc);
    infiniStatus_t quant(
        void *workspace,
        size_t workspace_size,
        void *packed_weights,
        void *b_scale,
        void *zero,
        const void *a,
        const void *b,
        void *stream) const;
    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        const void *a,
        void *packed_weights,
        void *b_scale,
        void *zero,
        void *stream) const;
};
} // namespace op::matmul_gptq::cpu

#endif // __MATMUL_GPTQ_CPU_H__
