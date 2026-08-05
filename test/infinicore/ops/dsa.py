import torch
from ixtriturbo._C import ops as ixops
from vllm_iluvatar.custom_kernels.compute_block_sparse_mqa_logits.perf_impl import (
    compute_block_sparse_mqa_logits_perf,
)
from vllm_iluvatar.custom_kernels.fused_deepseek_v2_indexer_postprocess.perf_impl import (
    fused_deepseek_v2_indexer_postprocess_perf,
)
from vllm_iluvatar.custom_kernels.indexer_k_cache.perf_impl import (
    indexer_k_cache_perf,
)
from vllm_iluvatar.custom_kernels.map_prefill_request_block_indices_to_global_blocks.perf_impl import (
    map_decode_request_block_indices_to_global_blocks_perf,
    map_prefill_request_block_indices_to_global_blocks_perf,
)
from vllm_iluvatar.custom_kernels.select_prefill_topk_block_indices.perf_impl import (
    select_decode_topk_block_indices_perf,
    select_prefill_topk_block_indices_perf,
)

import infinicore


def ic(t):
    return infinicore.from_torch(t)


dev = "cuda"

# block sparse MQA logits
q = torch.randn(4, 64, 128, device=dev, dtype=torch.bfloat16)
cache = torch.randn(2, 64, 128, device=dev, dtype=torch.bfloat16)
cuq = torch.tensor([0, 4], device=dev, dtype=torch.int32)
cukv = torch.tensor([0, 4], device=dev, dtype=torch.int32)
blocks = torch.tensor([[0]], device=dev, dtype=torch.int32)
weights = torch.randn(4, 64, device=dev, dtype=torch.bfloat16)
a = torch.zeros(4, 64, device=dev, dtype=torch.float32)
b = torch.zeros_like(a)
compute_block_sparse_mqa_logits_perf(q, cuq, cukv, cache, blocks, weights, a, 4, 4, 64)
torch.cuda.synchronize()
infinicore.compute_block_sparse_mqa_logits_(
    ic(b), ic(q), ic(cache), ic(cuq), ic(cukv), ic(blocks), ic(weights), 4, 4, 64
)
infinicore.sync_device()
torch.testing.assert_close(a, b)
print("block_sparse_logits ok")

# prefill topk
ks = torch.tensor([0, 0, 0, 0], device=dev, dtype=torch.int32)
ke = torch.tensor([1, 2, 3, 4], device=dev, dtype=torch.int32)
a = torch.empty(4, 8, device=dev, dtype=torch.int32)
b = torch.empty_like(a)
select_prefill_topk_block_indices_perf(
    a.new_empty((4, 64), dtype=torch.float32).normal_(), ks, ke, a
)
# share identical logits for comparison
logits = torch.randn(4, 64, device=dev)
select_prefill_topk_block_indices_perf(logits, ks, ke, a)
torch.cuda.synchronize()
infinicore.select_prefill_topk_block_indices_(ic(b), ic(logits), ic(ks), ic(ke))
infinicore.sync_device()
torch.testing.assert_close(a, b)
print("select_prefill_topk ok")

# prefill mapping without workspace
req = torch.tensor([0, 0, 1, 1], device=dev, dtype=torch.int32)
bt = torch.tensor([[2, 3], [5, 6]], device=dev, dtype=torch.int32)
idx = torch.tensor(
    [[0, 63, 64, -1], [1, 65, -1, -1], [0, 64, -1, -1], [3, 66, -1, -1]],
    device=dev,
    dtype=torch.int32,
)
a = map_prefill_request_block_indices_to_global_blocks_perf(req, bt, idx, 64)
b = torch.empty_like(idx)
torch.cuda.synchronize()
infinicore.map_prefill_request_block_indices_(ic(b), ic(req), ic(bt), ic(idx), 64)
infinicore.sync_device()
torch.testing.assert_close(a, b)
print("map_prefill ok")

# fused indexer postprocess
T, H, D, R = 3, 64, 128, 64
q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
kw = torch.randn(T, D + H, device=dev, dtype=torch.bfloat16)
nw = torch.randn(D, device=dev, dtype=torch.bfloat16)
nb = torch.randn(D, device=dev, dtype=torch.bfloat16)
pos = torch.arange(T, device=dev, dtype=torch.int64)
cs = torch.randn(16, R, device=dev, dtype=torch.bfloat16)
slots = torch.tensor([0, 65, 130], device=dev, dtype=torch.int64)
qa = torch.empty_like(q)
qb = torch.empty_like(q)
ka = torch.empty(0, D, device=dev, dtype=torch.bfloat16)
kb = torch.empty_like(ka)
wa = torch.empty(T, H, device=dev, dtype=torch.bfloat16)
wb = torch.empty_like(wa)
ca = torch.zeros(4, 64, D, device=dev, dtype=torch.bfloat16)
cb = ca.clone()
fused_deepseek_v2_indexer_postprocess_perf(
    qa, ka, wa, ca, slots, q, kw, nw, nb, pos, cs, T, False, 1e-6, 0.01
)
torch.cuda.synchronize()
infinicore.fused_deepseek_v2_indexer_postprocess_(
    ic(qb),
    ic(kb),
    ic(wb),
    ic(cb),
    ic(slots),
    ic(q),
    ic(kw),
    ic(nw),
    ic(nb),
    ic(pos),
    ic(cs),
    T,
    False,
    1e-6,
    0.01,
)
infinicore.sync_device()
torch.testing.assert_close(qa, qb)
infinicore.sync_device()
torch.testing.assert_close(wa, wb)
infinicore.sync_device()
torch.testing.assert_close(ca, cb)
print("fused_indexer_postprocess ok")

# production ixtriturbo sparse MLA v2
T, H, D, V, K, N = 1, 64, 576, 512, 64, 64
query = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
kv = torch.randn(N, 1, D, device=dev, dtype=torch.bfloat16)
indices = torch.arange(K, device=dev, dtype=torch.int32).view(T, 1, K)
lens = torch.tensor([K], device=dev, dtype=torch.int32)
a = torch.empty(T, H, V, device=dev, dtype=torch.bfloat16)
b = torch.empty_like(a)
ixops.flash_mla_sparse_v2(a, query, kv, indices, lens, float(D**-0.5), None)
torch.cuda.synchronize()
infinicore.sparse_flash_mla_(
    ic(b), ic(query), ic(kv), ic(indices), ic(lens), float(D**-0.5)
)
infinicore.sync_device()
torch.testing.assert_close(a, b)
print("sparse_flash_mla_v2 ok")


# indexer cache
keys = torch.randn(3, 128, device=dev, dtype=torch.bfloat16)
slots = torch.tensor([0, 65, 130], device=dev, dtype=torch.int64)
cache_ref = torch.zeros(4, 64, 128, device=dev, dtype=torch.bfloat16)
cache_out = cache_ref.clone()
indexer_k_cache_perf(keys, cache_ref, slots)
torch.cuda.synchronize()
infinicore.indexer_k_cache_(ic(keys), ic(cache_out), ic(slots))
infinicore.sync_device()
torch.testing.assert_close(cache_ref, cache_out)
print("indexer_k_cache ok")

# decode topk
logits = torch.randn(2, 128, device=dev, dtype=torch.float32)
seq_lens = torch.tensor([64, 96], device=dev, dtype=torch.int32)
topk_ref = torch.empty(2, 16, device=dev, dtype=torch.int32)
topk_out = torch.empty_like(topk_ref)
select_decode_topk_block_indices_perf(logits, seq_lens, topk_ref)
torch.cuda.synchronize()
infinicore.select_decode_topk_block_indices_(ic(topk_out), ic(logits), ic(seq_lens))
infinicore.sync_device()
torch.testing.assert_close(topk_ref, topk_out)
print("select_decode_topk ok")

# decode mapping
req = torch.tensor([0, 1], device=dev, dtype=torch.int32)
block_table = torch.tensor([[2, 3], [5, 6]], device=dev, dtype=torch.int32)
token_indices = torch.tensor(
    [[0, 63, 64, -1], [3, 66, -1, -1]],
    device=dev,
    dtype=torch.int32,
)
mapped_ref = map_decode_request_block_indices_to_global_blocks_perf(
    req, block_table, token_indices, 64
)
mapped_out = torch.empty_like(token_indices)
torch.cuda.synchronize()
infinicore.map_decode_request_block_indices_(
    ic(mapped_out), ic(req), ic(block_table), ic(token_indices), 64
)
infinicore.sync_device()
torch.testing.assert_close(mapped_ref, mapped_out)
print("map_decode ok")

# valid sparse topk lengths
indices = torch.tensor(
    [[[1, 2, -1, -1]], [[3, 4, 5, -1]]],
    device=dev,
    dtype=torch.int32,
)
lens_ref = torch.empty(2, device=dev, dtype=torch.int32)
lens_out = torch.empty_like(lens_ref)
ixops.topk_indices_context_lens(lens_ref, indices)
torch.cuda.synchronize()
infinicore.topk_indices_context_lens_(ic(lens_out), ic(indices))
infinicore.sync_device()
torch.testing.assert_close(lens_ref, lens_out)
print("topk_indices_context_lens ok")

# InfiniCore graph record/replay keeps the bridge on the current InfiniCore stream.
graph_cache = torch.zeros_like(cache_ref)
torch.cuda.synchronize()
infinicore.start_graph_recording()
infinicore.indexer_k_cache_(ic(keys), ic(graph_cache), ic(slots))
graph = infinicore.stop_graph_recording()
graph.run()
infinicore.sync_device()
torch.testing.assert_close(cache_ref, graph_cache)
print("graph_replay ok")
