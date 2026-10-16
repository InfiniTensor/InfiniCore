#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/hygon_lightop_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::dsa_impl::hygon {

void select_prefill_topk(Tensor topk_indices, const Tensor &logits,
                         const Tensor &cu_seqlen_ks,
                         const Tensor &cu_seqlen_ke) {
  adaptor::set_aten_stream_to_infinicore();
  if (!adaptor::hygon_lightop::top_k_per_row_available()) {
    throw std::runtime_error(
        "select_prefill_topk_block_indices requires LightOp on Hygon");
  }
  auto logits_at = adaptor::to_aten_tensor(logits);
  auto cu_seqlen_ks_at = adaptor::to_aten_tensor(cu_seqlen_ks);
  auto cu_seqlen_ke_at = adaptor::to_aten_tensor(cu_seqlen_ke);
  auto topk_indices_at = adaptor::to_aten_tensor(topk_indices);
  topk_indices_at.fill_(-1);
  adaptor::hygon_lightop::top_k_per_row_prefill(
      logits_at, cu_seqlen_ks_at, cu_seqlen_ke_at, topk_indices_at);
}

void select_decode_topk(Tensor topk_indices, const Tensor &logits,
                        const Tensor &seq_lens) {
  adaptor::set_aten_stream_to_infinicore();
  if (!adaptor::hygon_lightop::top_k_per_row_available()) {
    throw std::runtime_error(
        "select_decode_topk_block_indices requires LightOp on Hygon");
  }
  auto logits_at = adaptor::to_aten_tensor(logits);
  auto seq_lens_at = adaptor::to_aten_tensor(seq_lens);
  auto topk_indices_at = adaptor::to_aten_tensor(topk_indices);
  topk_indices_at.fill_(-1);
  adaptor::hygon_lightop::top_k_per_row_decode(logits_at, seq_lens_at,
                                               topk_indices_at);
  topk_indices_at.masked_fill_(topk_indices_at.ge(seq_lens_at.unsqueeze(-1)),
                               -1);
}

void map_prefill_indices(Tensor output, const Tensor &req_id,
                         const Tensor &block_table, const Tensor &token_indices,
                         int64_t block_size, bool has_prefill_workspace,
                         std::optional<Tensor> prefill_workspace_request_ids,
                         std::optional<Tensor> prefill_workspace_starts) {
  adaptor::set_aten_stream_to_infinicore();
  if (!adaptor::hygon_lightop::convert_req_index_to_global_index_available()) {
    throw std::runtime_error(
        "map_prefill_request_block_indices requires LightOp on Hygon");
  }
  std::optional<at::Tensor> request_ids_at;
  std::optional<at::Tensor> starts_at;
  if (prefill_workspace_request_ids) {
    request_ids_at = adaptor::to_aten_tensor(*prefill_workspace_request_ids);
  }
  if (prefill_workspace_starts) {
    starts_at = adaptor::to_aten_tensor(*prefill_workspace_starts);
  }
  auto token_indices_at = adaptor::to_aten_tensor(token_indices);
  auto mapped = adaptor::hygon_lightop::convert_req_index_to_global_index(
      adaptor::to_aten_tensor(req_id), adaptor::to_aten_tensor(block_table),
      token_indices_at, static_cast<int>(block_size), has_prefill_workspace,
      std::move(request_ids_at), std::move(starts_at));
  // LightOp may use either -1 or its configured maximum sequence length as
  // the unused-slot sentinel. Normalize both forms before context lengths
  // are derived or an attention kernel consumes the mapped indices.
  const int64_t max_request_tokens = block_table->size(1) * block_size;
  auto invalid = token_indices_at.lt(0).logical_or(
      token_indices_at.ge(max_request_tokens));
  mapped.masked_fill_(invalid, -1);
  adaptor::to_aten_tensor(output).copy_(mapped);
}

void map_decode_indices(Tensor output, const Tensor &req_id,
                        const Tensor &block_table, const Tensor &token_indices,
                        int64_t block_size) {
  adaptor::set_aten_stream_to_infinicore();
  auto output_at = adaptor::to_aten_tensor(output);
  auto req_id_at = adaptor::to_aten_tensor(req_id);
  auto block_table_at = adaptor::to_aten_tensor(block_table);
  auto token_indices_at = adaptor::to_aten_tensor(token_indices);

  const int64_t max_request_tokens = block_table_at.size(1) * block_size;
  auto valid = token_indices_at.ge(0).logical_and(
      token_indices_at.lt(max_request_tokens));
  auto safe_indices = token_indices_at.clamp(0, max_request_tokens - 1);
  auto logical_blocks = at::floor_divide(safe_indices, block_size);
  auto block_offsets = at::remainder(safe_indices, block_size);
  auto request_blocks =
      block_table_at.index_select(0, req_id_at.to(at::ScalarType::Long));
  auto physical_blocks =
      request_blocks.gather(1, logical_blocks.to(at::ScalarType::Long));
  auto mapped = physical_blocks * block_size + block_offsets;
  mapped.masked_fill_(valid.logical_not(), -1);
  output_at.copy_(mapped);
}

void topk_context_lens(Tensor topk_lens, const Tensor &indices) {
  adaptor::set_aten_stream_to_infinicore();
  auto indices_at = adaptor::to_aten_tensor(indices);
  auto topk_lens_at = adaptor::to_aten_tensor(topk_lens);
  auto valid = indices_at.reshape({indices_at.size(0), -1}).ge(0);
  topk_lens_at.copy_(valid.sum({1}, false, at::ScalarType::Int));
}

static bool registered = []() {
  vendor_ops::select_prefill_topk_dispatcher().registerDevice(
      Device::Type::HYGON, &select_prefill_topk);
  vendor_ops::select_decode_topk_dispatcher().registerDevice(
      Device::Type::HYGON, &select_decode_topk);
  vendor_ops::map_prefill_indices_dispatcher().registerDevice(
      Device::Type::HYGON, &map_prefill_indices);
  vendor_ops::map_decode_indices_dispatcher().registerDevice(
      Device::Type::HYGON, &map_decode_indices);
  vendor_ops::topk_context_lens_dispatcher().registerDevice(Device::Type::HYGON,
                                                            &topk_context_lens);
  return true;
}();

} // namespace infinicore::op::dsa_impl::hygon

#endif
