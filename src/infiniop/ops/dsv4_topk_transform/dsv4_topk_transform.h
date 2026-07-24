#ifndef DSV4_TOPK_TRANSFORM_H
#define DSV4_TOPK_TRANSFORM_H
#include "../../operator.h"
#include "info.h"
#define DESCRIPTOR(NAMESPACE)                                                                                                                                                                                                                                            \
    namespace op::dsv4_topk_transform::NAMESPACE {                                                                                                                                                                                                                       \
    class Descriptor final : public InfiniopDescriptor {                                                                                                                                                                                                                 \
        Info _info;                                                                                                                                                                                                                                                      \
        size_t _workspace_size;                                                                                                                                                                                                                                          \
        Descriptor(Info info, size_t workspace_size, infiniDevice_t device_type, int device_id) : InfiniopDescriptor{device_type, device_id}, _info(info), _workspace_size(workspace_size) {}                                                                            \
                                                                                                                                                                                                                                                                         \
    public:                                                                                                                                                                                                                                                              \
        size_t workspaceSize() const { return _workspace_size; }                                                                                                                                                                                                         \
        const Info &info() const { return _info; }                                                                                                                                                                                                                       \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_tables_desc, int page_size); \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *out, const void *scores, const void *seq_lens, const void *page_tables, void *stream) const;                                                                                              \
    };                                                                                                                                                                                                                                                                   \
    }
#endif
