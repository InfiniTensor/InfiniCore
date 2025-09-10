#include "paged_attention_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <cmath>
#include <vector>

// TODO finish cpu version

namespace op::paged_attention::cpu {

Descriptor::~Descriptor() {}

// Factory function to create a CPU descriptor for Paged Attention.
// NOTE: This part is already well-structured and consistent with the CUDA version, so no changes are needed.
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    const std::optional<infiniopTensorDescriptor_t>& alibi_slopes_desc,
    float scale
    ) {

    // auto result = PagedAttentionInfo::create(out_desc, q_desc, k_cache_desc, v_cache_desc,
    //                                            block_tables_desc, seq_lens_desc, alibi_slopes_desc, scale);
    // CHECK_RESULT(result);
    // *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// =================================================================================
// MODIFIED: The core CPU logic is completely refactored below.
// =================================================================================
// template <typename T>
// infiniStatus_t paged_attention(const PagedAttentionInfo *info,
//                                   T *out, const T *q,
//                                   const T *k_cache, const T *v_cache,
//                                   const int *block_tables, const int *seq_lens,
//                                   const float *alibi_slopes) {
// Parallelize the operation over sequences and heads using OpenMP.
// #pragma omp parallel for
//     for (ptrdiff_t i = 0; i < ptrdiff_t(info->num_seqs * info->num_heads); ++i) {
//         const size_t seq_idx = i / info->num_heads;
//         const size_t head_idx = i % info->num_heads;
//         const size_t seq_len = seq_lens[seq_idx];
        
//         if (seq_len == 0) continue;

//         // MODIFIED: Pointer arithmetic now strictly uses strides from the info struct.
//         // We cast to char* to perform byte-level stride calculations, which is the safest way.
//         const char* q_base_ptr = (const char*)q + seq_idx * info->q_stride;
//         const T* q_vec = (const T*)(q_base_ptr) + head_idx * info->head_size;
        
//         char* out_base_ptr = (char*)out + seq_idx * info->q_stride; // Output has same layout as query
//         T* out_vec = (T*)(out_base_ptr) + head_idx * info->head_size;

//         const size_t kv_head_idx = head_idx / (info->num_heads / info->num_kv_heads);

//         std::vector<float> logits(seq_len);
//         float max_logit = -INFINITY;

//         // 1. Compute QK dot products and find max logit
//         for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
//             const size_t block_table_idx = seq_idx * info->max_num_blocks_per_seq + token_idx / info->block_size;
//             const size_t block_num = block_tables[block_table_idx];
//             const size_t block_off = token_idx % info->block_size;

//             // MODIFIED: K-Cache access logic now matches the CUDA kernel's high-performance layout.
//             // Layout assumption: [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_SIZE]
//             const char* k_block_ptr = (const char*)k_cache + block_num * info->kv_block_stride;
//             const char* k_head_ptr = k_block_ptr + kv_head_idx * info->kv_head_stride;
//             const T* k_vec_ptr = (const T*)k_head_ptr + block_off * info->head_size;

//             float qk = 0.0f;
//             for (size_t h = 0; h < info->head_size; ++h) {
//                 qk += utils::cast<float>(q_vec[h]) * utils::cast<float>(k_vec_ptr[h]);
//             }
            
//             logits[token_idx] = qk * info->scale;
//             if (info->has_alibi) {
//                 logits[token_idx] += alibi_slopes[head_idx] * (token_idx - seq_len + 1);
//             }
//             if (logits[token_idx] > max_logit) {
//                 max_logit = logits[token_idx];
//             }
//         }

//         // 2. Compute Softmax
//         float exp_sum = 0.0f;
//         for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
//             float val = std::exp(logits[token_idx] - max_logit);
//             logits[token_idx] = val;
//             exp_sum += val;
//         }

//         const float inv_sum = 1.0f / (exp_sum + 1e-6f);
//         for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
//             logits[token_idx] *= inv_sum;
//         }

//         // 3. Aggregate V values
//         std::vector<float> acc(info->head_size, 0.0f);
//         for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
//             const size_t block_table_idx = seq_idx * info->max_num_blocks_per_seq + token_idx / info->block_size;
//             const size_t block_num = block_tables[block_table_idx];
//             const size_t block_off = token_idx % info->block_size;
//             const float prob = logits[token_idx];
            
//             // MODIFIED: V-Cache access logic also matches the CUDA kernel's layout.
//             // We assume K and V have the same layout and strides.
//             const char* v_block_ptr = (const char*)v_cache + block_num * info->kv_block_stride;
//             const char* v_head_ptr = v_block_ptr + kv_head_idx * info->kv_head_stride;
//             const T* v_vec_ptr = (const T*)v_head_ptr + block_off * info->head_size;
            
//             for (size_t h = 0; h < info->head_size; ++h) {
//                 acc[h] += prob * utils::cast<float>(v_vec_ptr[h]);
//             }
//         }

//         for(size_t h = 0; h < info->head_size; ++h) {
//             out_vec[h] = utils::cast<T>(acc[h]);
//         }
//     }
//     return INFINI_STATUS_SUCCESS;
// }

// Dispatches the call to the correct templated implementation based on dtype.
// NOTE: This part is also consistent, no changes needed.
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables, const void *seq_lens, const void *alibi_slopes,
    void *stream) const {

    // // NOTE: CPU version typically uses F32 for computation. If F16/BF16 support
    // // is needed, conversions or specialized libraries would be required.
    // if (_info.dtype == INFINI_DTYPE_F32) {
    //     CHECK_STATUS(paged_attention<float>(&_info, (float *)out, (const float *)q, (const float *)k_cache,
    //                                            (const float *)v_cache, (const int *)block_tables,
    //                                            (const int *)seq_lens, (const float *)alibi_slopes));
    // } else {
    //     return INFINI_STATUS_BAD_TENSOR_DTYPE;
    // }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::paged_attention::cpu
