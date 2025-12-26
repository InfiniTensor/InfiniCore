import torch
import ctypes
from ctypes import c_uint64
import math
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    TestWorkspace,
)

# ==============================================================================
# 模拟上层调度器 (与 PagedCaching 对齐)
# ==============================================================================
class SimpleCacheManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.request_to_blocks = {}
        self.request_to_len = {}

    def allocate_slots(self, request_id, num_new_tokens):
        if request_id not in self.request_to_len:
            self.request_to_len[request_id] = 0
            self.request_to_blocks[request_id] = []
        
        start_pos = self.request_to_len[request_id]
        new_total_len = start_pos + num_new_tokens
        needed_blocks = (new_total_len + self.block_size - 1) // self.block_size
        added_blocks = needed_blocks - len(self.request_to_blocks[request_id])
        
        for _ in range(added_blocks):
            self.request_to_blocks[request_id].append(self.free_blocks.pop(0))
            
        self.request_to_len[request_id] = new_total_len
        # 返回当前该 Request 的完整 block_table
        return self.request_to_blocks[request_id], new_total_len

# ==============================================================================
# Reference Implementation (支持多轮增量 Query)
# ==============================================================================
def ref_paged_attention_multi_turn(
    query_new, k_cache, v_cache, block_tables, seq_lens, scale, alibi_slopes
):
    """
    Reference: 计算新 Query 对全量 KV 的注意力
    query_new: [batch, n_new_tokens, n_heads, dh]
    seq_lens: [batch] 包含每个 seq 的总长度 (history + new)
    """
    batch_size = query_new.shape[0]
    num_heads = query_new.shape[2]
    head_size = k_cache.shape[3]
    block_size = k_cache.shape[2]
    
    outputs = []
    
    for i in range(batch_size):
        total_len = seq_lens[i].item()
        num_new = query_new.shape[1]
        history_len = total_len - num_new
        
        # 1. 提取全量 KV (从 block_table 中 gather)
        table = block_tables[i]
        keys_all = []
        values_all = []
        for j in range(total_len):
            b_id = table[j // block_size].item()
            off = j % block_size
            keys_all.append(k_cache[b_id, :, off, :])
            values_all.append(v_cache[b_id, :, off, :])
        
        K = torch.stack(keys_all, dim=0) # [total_len, n_kv_heads, dh]
        V = torch.stack(values_all, dim=0)
        Q = query_new[i] # [num_new, n_heads, dh]

        # 2. 计算 Attention Score
        # [num_new, n_heads, dh] * [total_len, n_kv_heads, dh] -> [n_heads, num_new, total_len]
        scores = torch.einsum("qhd,khd->hqk", Q, K).float() * scale
        
        # 3. 构造因果掩码 (Causal Mask)
        # 新 Query 只能看到之前的 Token 和当前及之前的自己
        mask = torch.full((num_new, total_len), float("-inf"), device=Q.device)
        for q_idx in range(num_new):
            # 逻辑位置 = history_len + q_idx
            mask[q_idx, : history_len + q_idx + 1] = 0.0
        
        scores = scores + mask.unsqueeze(0)
        attn_weights = torch.softmax(scores, dim=-1).to(Q.dtype)
        
        # 4. 加权求和
        out = torch.einsum("hqk,khd->qhd", attn_weights, V)
        outputs.append(out)
        
    return torch.stack(outputs, dim=0)

# ==============================================================================
# Test Operator 实现
# ==============================================================================
def test(
    handle,
    device,
    num_seqs,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    max_step_len, # 每一轮新增长度上限
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(f"Testing Multi-turn PagedAttention | num_seqs={num_seqs}, dtype={InfiniDtypeNames[dtype]}")

    num_blocks = 8192
    manager = SimpleCacheManager(num_blocks, block_size)
    scale =  head_size ** -0.5
    
    # 初始化全局物理缓存 (持久化)
    k_cache = TestTensor((num_blocks, num_kv_heads, block_size, head_size), None, dtype, device)
    v_cache = TestTensor((num_blocks, num_kv_heads, block_size, head_size), None, dtype, device)
    
    # 模拟两轮对话
    num_rounds = 2
    for r in range(num_rounds):
        print(f"--- Round {r+1} ---")
        
        # 1. 模拟调度：确定本轮每个 seq 新增多少 token，并获取最新的 block_table
        new_lens = torch.randint(1, max_step_len + 1, (num_seqs,), dtype=torch.int32)
        total_lens_list = []
        max_blocks_needed = 0
        
        # 收集 block_tables
        all_block_tables = []
        for i in range(num_seqs):
            table, total_len = manager.allocate_slots(i, new_lens[i].item())
            total_lens_list.append(total_len)
            all_block_tables.append(table)
            max_blocks_needed = max(max_blocks_needed, len(table))
            
        # 对齐 block_tables (padding to max_blocks_needed)
        padded_tables = []
        for table in all_block_tables:
            padded_table = table + [0] * (max_blocks_needed - len(table))
            padded_tables.append(padded_table)
            
        # 准备算子输入张量
        max_new_len = new_lens.max().item()
        # 注意：此处的 q 通常为 [batch, num_new, n_heads, dh]
        q_new_torch = torch.randn(num_seqs, max_new_len, num_heads, head_size)
        
        q_new = TestTensor.from_torch(q_new_torch, dtype, device)
        out = TestTensor((num_seqs, max_new_len, num_heads, head_size), None, dtype, device)
        
        seq_lens_torch = torch.tensor(total_lens_list, dtype=torch.int32)
        seq_lens = TestTensor.from_torch(seq_lens_torch, InfiniDtype.I32, device)
        
        block_tables = TestTensor.from_torch(torch.tensor(padded_tables, dtype=torch.int32), InfiniDtype.I32, device)

        # 模拟：在调用 Attention 前，KV 已经通过 PagedCaching 写入了缓存
        # (此处省略 PagedCaching 调用，直接用相同的逻辑更新 reference 中的虚拟缓存)
        
        # 2. Reference 计算
        ans = ref_paged_attention_multi_turn(
            q_new.torch_tensor(), k_cache.torch_tensor(), v_cache.torch_tensor(),
            block_tables.torch_tensor(), seq_lens.torch_tensor(), scale, None
        )

        if sync: sync()

        # 3. 创建描述符并执行
        descriptor = infiniopOperatorDescriptor_t()
        check_error(LIBINFINIOP.infiniopCreatePagedAttentionPrefillDescriptor(
            handle, ctypes.byref(descriptor),
            out.descriptor, q_new.descriptor,
            k_cache.descriptor, v_cache.descriptor,
            block_tables.descriptor, seq_lens.descriptor,
            None, scale # ALiBi slopes 传空
        ))

        workspace_size = c_uint64(0)
        check_error(LIBINFINIOP.infiniopGetPagedAttentionPrefillWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
        workspace = TestWorkspace(workspace_size.value, device)

        # 执行算子
        check_error(LIBINFINIOP.infiniopPagedAttentionPrefill(
            descriptor, workspace.data(), workspace_size.value,
            out.data(), q_new.data(),
            k_cache.data(), v_cache.data(),
            block_tables.data(), seq_lens.data(),
            None, None
        ))

        if sync: sync()

        # 4. 验证
        atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
        assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)
        
        check_error(LIBINFINIOP.infiniopDestroyPagedAttentionPrefillDescriptor(descriptor))
        print(f"Round {r+1} verified.")

# ==============================================================================
# 配置
# ==============================================================================
_TEST_CASES_ = [
    # (num_seqs, num_heads, num_kv_heads, head_size, block_size, max_step_len)
    (2, 8, 8, 128, 16, 64),
    (4, 32, 32, 128, 16, 128),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
}

if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)
    print("\033[92mMulti-turn PagedAttention test passed!\033[0m")
