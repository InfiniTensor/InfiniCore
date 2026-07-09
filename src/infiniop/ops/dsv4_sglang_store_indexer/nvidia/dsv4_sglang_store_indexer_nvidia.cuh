#ifndef DSV4_SGLANG_STORE_INDEXER_NVIDIA_CUH
#define DSV4_SGLANG_STORE_INDEXER_NVIDIA_CUH
#include "../dsv4_sglang_store_indexer.h"
namespace op::dsv4_sglang_store_indexer::nvidia {
class Descriptor final : public op::dsv4_sglang_store_indexer::Descriptor {
public:
    using op::dsv4_sglang_store_indexer::Descriptor::Descriptor;
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t cache_desc, infiniopTensorDescriptor_t indices_desc);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, const void *input, void *cache, const void *indices, void *stream) const;
};
} // namespace op::dsv4_sglang_store_indexer::nvidia
#endif
