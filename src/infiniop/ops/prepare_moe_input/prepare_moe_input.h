#ifndef __PREPARE_MOE_INPUT_H__
#define __PREPARE_MOE_INPUT_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                      \
                                                                                                                   \
    namespace op::prepare_moe_input::NAMESPACE {                                                                   \
    class Descriptor final : public InfiniopDescriptor {                                                           \
        struct Opaque;                                                                                             \
        Opaque *_opaque;                                                                                           \
        PrepareMoeInputInfo _info;                                                                                 \
        size_t _workspace_size;                                                                                    \
                                                                                                                   \
        Descriptor(Opaque *opaque, PrepareMoeInputInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}     \
                                                                                                                   \
    public:                                                                                                        \
        ~Descriptor();                                                                                             \
                                                                                                                   \
        size_t workspaceSize() const { return _workspace_size; }                                                   \
                                                                                                                   \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,                               \
                                     infiniopTensorDescriptor_t expert_offsets_desc,                               \
                                     infiniopTensorDescriptor_t blockscale_offsets_desc,                           \
                                     infiniopTensorDescriptor_t problem_sizes1_desc,                               \
                                     infiniopTensorDescriptor_t problem_sizes2_desc,                               \
                                     infiniopTensorDescriptor_t input_permutation_desc,                            \
                                     infiniopTensorDescriptor_t output_permutation_desc,                           \
                                     infiniopTensorDescriptor_t topk_ids_desc,                                     \
                                     size_t num_experts, size_t n, size_t k);                                      \
                                                                                                                   \
        infiniStatus_t calculate(void *workspace, size_t workspace_size,                                           \
                                 void *expert_offsets, void *blockscale_offsets,                                   \
                                 void *problem_sizes1, void *problem_sizes2,                                       \
                                 void *input_permutation, void *output_permutation,                                \
                                 const void *topk_ids, void *stream) const;                                        \
    };                                                                                                             \
    }

#endif
