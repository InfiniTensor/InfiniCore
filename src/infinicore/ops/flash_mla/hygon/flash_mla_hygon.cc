#if defined(ENABLE_HYGON_API)

#include "infinicore/ops/flash_mla.hpp"

#include "infinicore/context/context.hpp"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace infinicore::op::flash_mla_impl::hygon {
namespace {

constexpr float LOG_2_E = 1.4426950408889634f;

enum class ModelType {
  V32,
  MODEL1,
};

struct alignas(32) DecodingSchedMeta {
  int begin_req_idx;
  int end_req_idx;
  int begin_block_idx;
  int end_block_idx;
  int begin_split_idx;
  int is_first_req_splitted;
  int is_last_req_splitted;
  int pad;
};

struct SparseAttnFwdParams {
  int s_q;
  int s_kv;
  int h_q;
  int h_kv;
  int d_qk;
  int d_v;
  int topk;
  float sm_scale;
  float sm_scale_div_log2;
  void *q;
  void *kv;
  int *indices;
  float *attn_sink;
  int *topk_length;
  int stride_q_s_q;
  int stride_q_h_q;
  int stride_kv_s_kv;
  int stride_kv_h_kv;
  int stride_indices_s_q;
  int stride_indices_h_kv;
  void *out;
  float *max_logits;
  float *lse;
  int num_sm;
  void *stream;
  // FlashMLA 1.2 extends the public parameter ABI with optional split-KV
  // controls. The GLM prefill path does not use them, but the vendor launcher
  // copies the complete 208-byte object into the HIP kernel argument buffer.
  // Keep the appended fields zero until they are needed by a public API.
  std::array<std::byte, 64> optional_controls{};
};
static_assert(sizeof(SparseAttnFwdParams) == 208);

struct SparseAttnDecodeParams {
  int b;
  int s_q;
  int h_q;
  int h_kv;
  int d_qk;
  int d_v;
  float sm_scale;
  float sm_scale_div_log2;
  int num_blocks;
  int page_block_size;
  int topk;
  ModelType model_type;
  void *q;
  void *kv;
  int *indices;
  int *topk_length;
  float *attn_sink;
  float *lse;
  void *out;
  int extra_num_blocks;
  int extra_page_block_size;
  int extra_topk;
  void *extra_kv;
  int *extra_indices;
  int *extra_topk_length;
  int stride_q_b;
  int stride_q_s_q;
  int stride_q_h_q;
  int stride_kv_block;
  int stride_kv_row;
  int stride_indices_b;
  int stride_indices_s_q;
  int stride_lse_b;
  int stride_lse_s_q;
  int stride_o_b;
  int stride_o_s_q;
  int stride_o_h_q;
  int stride_extra_kv_block;
  int stride_extra_kv_row;
  int stride_extra_indices_b;
  int stride_extra_indices_s_q;
  void *stream;
  float *lse_accum;
  float *o_accum;
  int stride_lse_accum_split;
  int stride_lse_accum_s_q;
  int stride_o_accum_split;
  int stride_o_accum_s_q;
  int stride_o_accum_h_q;
  DecodingSchedMeta *tile_scheduler_metadata_ptr;
  int *num_splits_ptr;
  int num_sm_parts;
};

struct GetDecodeSchedMetaParams {
  int b;
  int s_q;
  int block_size_n;
  int fixed_overhead_num_blocks;
  int topk;
  int extra_topk;
  int *topk_length;
  int *extra_topk_length;
  int *seqlens_k_ptr;
  DecodingSchedMeta *tile_scheduler_metadata_ptr;
  int *num_splits_ptr;
  int num_sm_parts;
  void *stream;
};

struct DenseAttnDecodeParams {
    using index_t = int64_t;

    int b;
    int s_q;
    int q_seq_per_hk;
    int d;
    int d_v;
    int h_q;
    int h_k;
    int num_blocks;
    int q_head_per_hk;
    bool is_causal;
    float scale_softmax;
    float scale_softmax_log2;
    void *q_ptr;
    void *k_ptr;
    void *o_ptr;
    float *softmax_lse_ptr;
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;
    int *block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int *seqlens_k_ptr;
    DecodingSchedMeta *tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *num_splits_ptr;
    int total_num_splits;
    float *softmax_lseaccum_ptr;
    float *oaccum_ptr;
    void *stream;
    bool use_split_kv;
    int partition_block_nums;
    bool is_gfx928;
};

struct CombineParams {
    int b;
    int s_q;
    int h_q;
    int d_v;
    float *lse;
    void *out;
    int stride_lse_b;
    int stride_lse_s_q;
    int stride_o_b;
    int stride_o_s_q;
    int stride_o_h_q;
    float *lse_accum;
    float *o_accum;
    int stride_lse_accum_split;
    int stride_lse_accum_s_q;
    int stride_o_accum_split;
    int stride_o_accum_s_q;
    int stride_o_accum_h_q;
    DecodingSchedMeta *tile_scheduler_metadata_ptr;
    int *num_splits_ptr;
    int num_sm_parts;
    float *attn_sink;
    void *stream;
    bool use_split_kv;
    int num_splits;
    int *seqlens_k_ptr;
    int partition_block_nums;
};

using sparse_prefill_fn = void (*)(const SparseAttnFwdParams &);
using decode_sched_fn = void (*)(GetDecodeSchedMetaParams &);
using sparse_decode_fn = void (*)(const SparseAttnDecodeParams &);
using dense_decode_fn = void (*)(DenseAttnDecodeParams &);
using dense_combine_fn = void (*)(CombineParams &);

struct Symbols {
  void *handle{nullptr};
  sparse_prefill_fn sparse_prefill{nullptr};
  decode_sched_fn decode_sched_no_split{nullptr};
    decode_sched_fn decode_sched{nullptr};
  sparse_decode_fn sparse_decode_h16{nullptr};
  sparse_decode_fn sparse_decode_h64{nullptr};
  sparse_decode_fn sparse_decode_h128{nullptr};
    dense_decode_fn dense_decode_bf16{nullptr};
    dense_decode_fn dense_decode_f16{nullptr};
    dense_combine_fn dense_combine_bf16{nullptr};
    dense_combine_fn dense_combine_f16{nullptr};
  std::string error;
};

template <typename Fn>
Fn load_symbol(void *handle, const char *name, std::string &error) {
  dlerror();
  auto result = reinterpret_cast<Fn>(dlsym(handle, name));
  if (const char *message = dlerror()) {
    error = message;
    return nullptr;
  }
  return result;
}

Symbols &symbols() {
  static Symbols value;
  static std::once_flag once;
  std::call_once(once, []() {
    const char *env_path = std::getenv("INFINICORE_HYGON_FLASH_MLA_SO");
    const std::array<const char *, 4> candidates{
        env_path,
        "hygon_flash_mla.so",
        "/usr/local/lib/python3.10/dist-packages/flash_mla/"
        "cuda.cpython-310-x86_64-linux-gnu.so",
        "/usr/local/lib/python3.12/site-packages/flash_mla/"
        "cuda.cpython-312-x86_64-linux-gnu.so",
    };
    for (const char *path : candidates) {
      if (path == nullptr || path[0] == '\0') {
        continue;
      }
      dlerror();
      void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
      if (handle == nullptr) {
        if (const char *message = dlerror()) {
          value.error = std::string(path) + ": " + message;
        }
        continue;
      }
      std::string error;
      auto sparse_prefill = load_symbol<sparse_prefill_fn>(
          handle, "_ZN5gfx9314run_fwd_kernelERK19SparseAttnFwdParams", error);
      auto decode_sched_no_split = load_symbol<decode_sched_fn>(
          handle,
          "_ZN4gfx96decode43run_get_decoding_sched_meta_no_split_"
          "kernelER24GetDecodeSchedMetaParams",
          error);
            auto decode_sched = load_symbol<decode_sched_fn>(
                handle,
                "_ZN4gfx96decode34run_get_decoding_sched_meta_kernelER24GetDecodeSchedMetaParams",
                error);
      auto sparse_decode_h16 = load_symbol<sparse_decode_fn>(
          handle,
          "_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_"
          "kernelIL9ModelType0ELi16EEEvRK22SparseAttnDecodeParams",
          error);
      auto sparse_decode_h64 = load_symbol<sparse_decode_fn>(
          handle,
          "_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_"
          "kernelIL9ModelType0ELi64EEEvRK22SparseAttnDecodeParams",
          error);
      auto sparse_decode_h128 = load_symbol<sparse_decode_fn>(
          handle,
          "_ZN5gfx936decode10sparse_fp839run_flash_splitkv_mla_fp8_sparse_"
          "kernelIL9ModelType0ELi128EEEvRK22SparseAttnDecodeParams",
          error);
            auto dense_decode_bf16 = load_symbol<dense_decode_fn>(
                handle,
                "_ZN5gfx9328run_flash_splitkv_mla_kernelIN7cutlass10bfloat16_tEEEvR21DenseAttnDecodeParams",
                error);
            auto dense_decode_f16 = load_symbol<dense_decode_fn>(
                handle,
                "_ZN5gfx9328run_flash_splitkv_mla_kernelIN7cutlass6half_tEEEvR21DenseAttnDecodeParams",
                error);
            auto dense_combine_bf16 = load_symbol<dense_combine_fn>(
                handle,
                "_ZN4gfx96decode28run_flash_mla_combine_kernelIN7cutlass10bfloat16_tEEEvR13CombineParams",
                error);
            auto dense_combine_f16 = load_symbol<dense_combine_fn>(
                handle,
                "_ZN4gfx96decode28run_flash_mla_combine_kernelIN7cutlass6half_tEEEvR13CombineParams",
                error);
            if (sparse_prefill == nullptr || decode_sched_no_split == nullptr || decode_sched == nullptr || sparse_decode_h16 == nullptr || sparse_decode_h64 == nullptr || sparse_decode_h128 == nullptr || dense_decode_bf16 == nullptr || dense_decode_f16 == nullptr || dense_combine_bf16 == nullptr || dense_combine_f16 == nullptr) {
        value.error = std::string(path) + ": " + error;
        dlclose(handle);
        continue;
      }
      value.handle = handle;
      value.sparse_prefill = sparse_prefill;
      value.decode_sched_no_split = decode_sched_no_split;
            value.decode_sched = decode_sched;
      value.sparse_decode_h16 = sparse_decode_h16;
      value.sparse_decode_h64 = sparse_decode_h64;
      value.sparse_decode_h128 = sparse_decode_h128;
            value.dense_decode_bf16 = dense_decode_bf16;
            value.dense_decode_f16 = dense_decode_f16;
            value.dense_combine_bf16 = dense_combine_bf16;
            value.dense_combine_f16 = dense_combine_f16;
      value.error.clear();
      return;
    }
  });
  return value;
}

int num_compute_units(const Device &device) {
  int current_device = 0;
  cudaDeviceProp properties{};
    if (cudaGetDevice(&current_device) != cudaSuccess || cudaGetDeviceProperties(&properties, current_device) != cudaSuccess) {
    throw std::runtime_error(
        "FlashMLA failed to query Hygon device properties");
  }
  if (static_cast<size_t>(current_device) != device.getIndex()) {
    throw std::runtime_error(
        "FlashMLA current device does not match tensor device");
  }
  return properties.multiProcessorCount;
}

int as_int(size_t value, const char *name) {
  if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("FlashMLA dimension exceeds int32: ") + name);
  }
  return static_cast<int>(value);
}

struct PrefillMeta {
  graph::GraphTensor output;
  graph::GraphTensor query;
  graph::GraphTensor kv_cache;
  graph::GraphTensor indices;
  graph::GraphTensor topk_lens;
  graph::GraphTensor max_logits;
  graph::GraphTensor lse;
  float scale;
  int num_sms;
};

void *plan_prefill(Tensor output, const Tensor &query, const Tensor &kv_cache,
                   const Tensor &indices, const Tensor &topk_lens,
                   float scale) {
  auto max_logits = Tensor::empty({query->size(0), query->size(1)},
                                  DataType::F32, query->device());
  auto lse = Tensor::empty({query->size(0), query->size(1)}, DataType::F32,
                           query->device());
  return new PrefillMeta{
      graph::GraphTensor(output),        graph::GraphTensor(query),
      graph::GraphTensor(kv_cache),      graph::GraphTensor(indices),
      graph::GraphTensor(topk_lens),     graph::GraphTensor(max_logits),
      graph::GraphTensor(lse),           scale,
      num_compute_units(query->device())};
}

void run_prefill(void *planned_meta) {
  auto *meta = reinterpret_cast<PrefillMeta *>(planned_meta);
  auto &api = symbols();
  if (api.sparse_prefill == nullptr) {
    throw std::runtime_error("Hygon FlashMLA is unavailable: " + api.error);
  }
  SparseAttnFwdParams params{
      as_int(meta->query->size(0), "s_q"),
      as_int(meta->kv_cache->size(0), "s_kv"),
      as_int(meta->query->size(1), "h_q"),
      as_int(meta->kv_cache->size(1), "h_kv"),
      as_int(meta->query->size(2), "d_qk"),
      as_int(meta->output->size(2), "d_v"),
      as_int(meta->indices->size(2), "topk"),
      meta->scale,
      meta->scale * LOG_2_E,
      meta->query->data(),
      meta->kv_cache->data(),
      reinterpret_cast<int *>(meta->indices->data()),
      nullptr,
      reinterpret_cast<int *>(meta->topk_lens->data()),
      as_int(meta->query->stride(0), "query stride 0"),
      as_int(meta->query->stride(1), "query stride 1"),
      as_int(meta->kv_cache->stride(0), "kv stride 0"),
      as_int(meta->kv_cache->stride(1), "kv stride 1"),
      as_int(meta->indices->stride(0), "indices stride 0"),
      as_int(meta->indices->stride(1), "indices stride 1"),
      meta->output->data(),
      reinterpret_cast<float *>(meta->max_logits->data()),
      reinterpret_cast<float *>(meta->lse->data()),
      meta->num_sms,
      context::getStream()};
  api.sparse_prefill(params);
}

struct DecodeMeta {
  graph::GraphTensor output;
  graph::GraphTensor query;
  graph::GraphTensor kv_cache;
  graph::GraphTensor indices;
  graph::GraphTensor topk_lens;
  graph::GraphTensor lse;
  graph::GraphTensor scheduler_metadata;
  graph::GraphTensor num_splits;
  float scale;
  int num_sm_parts;
};

void *plan_decode(Tensor output, const Tensor &query, const Tensor &kv_cache,
                  const Tensor &indices, const Tensor &topk_lens, float scale) {
  const int batch = as_int(query->size(0), "batch");
  const int heads = as_int(query->size(1), "heads");
  const int num_sm_parts = batch;
    auto lse = Tensor::empty({static_cast<size_t>(batch), 1, static_cast<size_t>(heads)},
                    DataType::F32, query->device());
    auto scheduler_metadata = Tensor::empty({static_cast<size_t>(num_sm_parts),
                     sizeof(DecodingSchedMeta) / sizeof(int)},
                    DataType::I32, query->device());
  auto num_splits = Tensor::empty({static_cast<size_t>(batch + 1)},
                                  DataType::I32, query->device());
  return new DecodeMeta{graph::GraphTensor(output),
                        graph::GraphTensor(query),
                        graph::GraphTensor(kv_cache),
                        graph::GraphTensor(indices),
                        graph::GraphTensor(topk_lens),
                        graph::GraphTensor(lse),
                        graph::GraphTensor(scheduler_metadata),
                        graph::GraphTensor(num_splits),
                        scale,
                        num_sm_parts};
}

void run_decode(void *planned_meta) {
  auto *meta = reinterpret_cast<DecodeMeta *>(planned_meta);
  auto &api = symbols();
  if (api.decode_sched_no_split == nullptr) {
    throw std::runtime_error("Hygon FlashMLA is unavailable: " + api.error);
  }
  const int batch = as_int(meta->query->size(0), "batch");
  const int heads = as_int(meta->query->size(1), "heads");
  const int topk = as_int(meta->indices->size(2), "topk");
  auto *stream = context::getStream();

  GetDecodeSchedMetaParams sched_params{
      batch,
      1,
      64,
      5,
      topk,
      0,
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<DecodingSchedMeta *>(meta->scheduler_metadata->data()),
      reinterpret_cast<int *>(meta->num_splits->data()),
      meta->num_sm_parts,
      stream};
  api.decode_sched_no_split(sched_params);

  SparseAttnDecodeParams params{
      batch,
      1,
      heads,
      1,
      as_int(meta->query->size(2), "d_qk"),
      as_int(meta->output->size(2), "d_v"),
      meta->scale,
      meta->scale * LOG_2_E,
      as_int(meta->kv_cache->size(0), "num_blocks"),
      as_int(meta->kv_cache->size(1), "page_block_size"),
      topk,
      ModelType::V32,
      meta->query->data(),
      meta->kv_cache->data(),
      reinterpret_cast<int *>(meta->indices->data()),
      nullptr,
      nullptr,
      reinterpret_cast<float *>(meta->lse->data()),
      meta->output->data(),
      0,
      0,
      0,
      nullptr,
      nullptr,
      nullptr,
      as_int(meta->query->stride(0), "query stride 0"),
      as_int(meta->query->stride(0), "query sequence stride"),
      as_int(meta->query->stride(1), "query stride 1"),
      as_int(meta->kv_cache->stride(0), "kv block stride"),
      as_int(meta->kv_cache->stride(1), "kv row stride"),
      as_int(meta->indices->stride(0), "indices stride 0"),
      as_int(meta->indices->stride(1), "indices stride 1"),
      as_int(meta->lse->stride(0), "lse stride 0"),
      as_int(meta->lse->stride(1), "lse stride 1"),
      as_int(meta->output->stride(0), "output stride 0"),
      as_int(meta->output->stride(0), "output sequence stride"),
      as_int(meta->output->stride(1), "output stride 1"),
      0,
      0,
      0,
      0,
      stream,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      0,
      reinterpret_cast<DecodingSchedMeta *>(meta->scheduler_metadata->data()),
      reinterpret_cast<int *>(meta->num_splits->data()),
      meta->num_sm_parts};
  if (heads <= 16) {
    api.sparse_decode_h16(params);
  } else if (heads == 64) {
    api.sparse_decode_h64(params);
  } else if (heads == 128) {
    api.sparse_decode_h128(params);
  } else {
    throw std::runtime_error(
        "Hygon FlashMLA sparse decode supports <=16, 64, or 128 query heads");
  }
}

struct DenseDecodeMeta {
    graph::GraphTensor output;
    graph::GraphTensor query;
    graph::GraphTensor kv_cache;
    graph::GraphTensor block_tables;
    graph::GraphTensor sequence_lengths;
    graph::GraphTensor lse;
    graph::GraphTensor lse_accum;
    graph::GraphTensor output_accum;
    graph::GraphTensor scheduler_metadata;
    graph::GraphTensor num_splits;
    float scale;
    int num_sm_parts;
};

void *plan_dense_decode(Tensor output, const Tensor &query,
                        const Tensor &kv_cache, const Tensor &block_tables,
                        const Tensor &sequence_lengths, float scale) {
    const size_t batch = query->size(0);
    const size_t heads = query->size(2);
    const size_t num_sms = static_cast<size_t>(num_compute_units(query->device()));
    const size_t num_sm_parts = std::max(num_sms / ((heads + 15) / 16), static_cast<size_t>(1));
    const size_t total_num_splits = batch + num_sm_parts;
    auto lse = Tensor::empty({batch, 1, heads}, DataType::F32, query->device());
    auto lse_accum = Tensor::empty({total_num_splits, 1, heads}, DataType::F32,
                                   query->device());
    auto output_accum = Tensor::empty(
        {total_num_splits, 1, heads, output->size(3)}, DataType::F32,
        query->device());
    auto scheduler_metadata = Tensor::empty(
        {num_sm_parts, sizeof(DecodingSchedMeta) / sizeof(int)}, DataType::I32,
        query->device());
    auto num_splits = Tensor::empty({batch + 1}, DataType::I32, query->device());
    return new DenseDecodeMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(query),
        graph::GraphTensor(kv_cache),
        graph::GraphTensor(block_tables),
        graph::GraphTensor(sequence_lengths),
        graph::GraphTensor(lse),
        graph::GraphTensor(lse_accum),
        graph::GraphTensor(output_accum),
        graph::GraphTensor(scheduler_metadata),
        graph::GraphTensor(num_splits),
        scale,
        as_int(num_sm_parts, "num_sm_parts")};
}

void run_dense_decode(void *planned_meta) {
    auto *meta = reinterpret_cast<DenseDecodeMeta *>(planned_meta);
    auto &api = symbols();
    if (api.decode_sched == nullptr || api.dense_decode_bf16 == nullptr || api.dense_decode_f16 == nullptr || api.dense_combine_bf16 == nullptr || api.dense_combine_f16 == nullptr) {
        throw std::runtime_error("Hygon dense FlashMLA is unavailable: " + api.error);
    }

    const int batch = as_int(meta->query->size(0), "batch");
    const int heads = as_int(meta->query->size(2), "heads");
    const int value_dim = as_int(meta->output->size(3), "value_dim");
    const int total_num_splits = batch + meta->num_sm_parts;
    auto *stream = context::getStream();

    GetDecodeSchedMetaParams sched_params{
        batch,
        1,
        64,
        5,
        -1,
        -1,
        nullptr,
        nullptr,
        reinterpret_cast<int *>(meta->sequence_lengths->data()),
        reinterpret_cast<DecodingSchedMeta *>(meta->scheduler_metadata->data()),
        reinterpret_cast<int *>(meta->num_splits->data()),
        meta->num_sm_parts,
        stream};
    api.decode_sched(sched_params);

    DenseAttnDecodeParams params{
        batch,
        1,
        heads,
        as_int(meta->query->size(3), "query_dim"),
        value_dim,
        heads,
        1,
        as_int(meta->kv_cache->size(0), "num_blocks"),
        heads,
        false,
        meta->scale,
        meta->scale * LOG_2_E,
        meta->query->data(),
        meta->kv_cache->data(),
        meta->output->data(),
        reinterpret_cast<float *>(meta->lse->data()),
        static_cast<int64_t>(meta->query->stride(0)),
        static_cast<int64_t>(meta->kv_cache->stride(0)),
        static_cast<int64_t>(meta->output->stride(0)),
        static_cast<int64_t>(meta->query->stride(2)),
        static_cast<int64_t>(meta->kv_cache->stride(1)),
        static_cast<int64_t>(meta->output->stride(2)),
        static_cast<int64_t>(meta->query->stride(1)),
        static_cast<int64_t>(meta->kv_cache->stride(2)),
        static_cast<int64_t>(meta->output->stride(1)),
        reinterpret_cast<int *>(meta->block_tables->data()),
        static_cast<int64_t>(meta->block_tables->stride(0)),
        as_int(meta->kv_cache->size(1), "page_block_size"),
        reinterpret_cast<int *>(meta->sequence_lengths->data()),
        reinterpret_cast<DecodingSchedMeta *>(meta->scheduler_metadata->data()),
        meta->num_sm_parts,
        reinterpret_cast<int *>(meta->num_splits->data()),
        total_num_splits,
        reinterpret_cast<float *>(meta->lse_accum->data()),
        reinterpret_cast<float *>(meta->output_accum->data()),
        stream,
        false,
        0,
        false};

    if (meta->query->dtype() == DataType::BF16) {
        api.dense_decode_bf16(params);
    } else {
        api.dense_decode_f16(params);
    }

    CombineParams combine_params{
        batch,
        1,
        heads,
        value_dim,
        reinterpret_cast<float *>(meta->lse->data()),
        meta->output->data(),
        heads,
        heads,
        heads * value_dim,
        heads * value_dim,
        value_dim,
        reinterpret_cast<float *>(meta->lse_accum->data()),
        reinterpret_cast<float *>(meta->output_accum->data()),
        heads,
        heads,
        heads * value_dim,
        heads * value_dim,
        value_dim,
        reinterpret_cast<DecodingSchedMeta *>(meta->scheduler_metadata->data()),
        reinterpret_cast<int *>(meta->num_splits->data()),
        meta->num_sm_parts,
        nullptr,
        stream,
        false,
        total_num_splits / batch,
        reinterpret_cast<int *>(meta->sequence_lengths->data()),
        0};
    if (meta->query->dtype() == DataType::BF16) {
        api.dense_combine_bf16(combine_params);
    } else {
        api.dense_combine_f16(combine_params);
    }
}

template <typename Meta>
void cleanup(void **planned_meta_ptr) {
  delete *reinterpret_cast<Meta **>(planned_meta_ptr);
  *planned_meta_ptr = nullptr;
}

static bool registered = []() {
  FlashMlaSparsePrefill::plan_dispatcher().registerDevice(Device::Type::HYGON,
                                                          &plan_prefill);
  FlashMlaSparsePrefill::run_dispatcher().registerDevice(Device::Type::HYGON,
                                                         &run_prefill);
  FlashMlaSparsePrefill::cleanup_dispatcher().registerDevice(
      Device::Type::HYGON, &cleanup<PrefillMeta>);
  FlashMlaSparseDecode::plan_dispatcher().registerDevice(Device::Type::HYGON,
                                                         &plan_decode);
  FlashMlaSparseDecode::run_dispatcher().registerDevice(Device::Type::HYGON,
                                                        &run_decode);
  FlashMlaSparseDecode::cleanup_dispatcher().registerDevice(
      Device::Type::HYGON, &cleanup<DecodeMeta>);
    FlashMlaDenseDecode::plan_dispatcher().registerDevice(
        Device::Type::HYGON, &plan_dense_decode);
    FlashMlaDenseDecode::run_dispatcher().registerDevice(
        Device::Type::HYGON, &run_dense_decode);
    FlashMlaDenseDecode::cleanup_dispatcher().registerDevice(
        Device::Type::HYGON, &cleanup<DenseDecodeMeta>);
  return true;
}();

} // namespace
} // namespace infinicore::op::flash_mla_impl::hygon

#endif
