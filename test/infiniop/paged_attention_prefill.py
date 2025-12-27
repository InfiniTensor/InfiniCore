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
# 模拟上层调度器 (与 PagedCaching 逻辑一致)
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
        return self.request_to_blocks[request_id], new_total_len

# ==============================================================================
# Reference 实现: 严格基于有效长度计算
# ==============================================================================
def ref_paged_attention_multi_turn(
    query_new, k_cache, v_cache, block_tables, seq_lens, new_lens, scale
):
    batch_size = query_new.shape[0]
    num_heads = query_new.shape[2]
    head_size = k_cache.shape[3]
    block_size = k_cache.shape[2]
    
    outputs = []
    for i in range(batch_size):
        total_len = seq_lens[i].item()
        num_new = new_lens[i].item() # 仅取本轮新增的部分
        history_len = total_len - num_new
        
        # 1. 提取全量有效 KV
        table = block_tables[i]
        keys_all, values_all = [], []
        for j in range(total_len):
            b_id = table[j // block_size].item()
            off = j % block_size
            keys_all.append(k_cache[b_id, :, off, :])
            values_all.append(v_cache[b_id, :, off, :])
        
        K = torch.stack(keys_all, dim=0) 
        V = torch.stack(values_all, dim=0)
        # 2. 仅提取有效的 Q (去除 Padding 影响)
        Q = query_new[i, :num_new, :, :] 

        # 3. 计算 Attention
        scores = torch.einsum("qhd,khd->hqk", Q, K).float() * scale
        
        # 4. 因果掩码 (Causal Mask)
        mask = torch.full((num_new, total_len), float("-inf"), device=Q.device)
        for q_idx in range(num_new):
            mask[q_idx, : history_len + q_idx + 1] = 0.0
        
        scores = scores + mask.unsqueeze(0)
        attn_weights = torch.softmax(scores, dim=-1).to(Q.dtype)
        
        out = torch.einsum("hqk,khd->qhd", attn_weights, V)
        
        # 为了与输出张量对齐，如果 Q 被 Padding 了，这里也补回 0
        padded_out = torch.zeros(query_new.shape[1], num_heads, head_size, device=Q.device, dtype=Q.dtype)
        padded_out[:num_new, :, :] = out
        outputs.append(padded_out)
        
    return torch.stack(outputs, dim=0)

# ==============================================================================
# Test Operator 实现
# ==============================================================================
def test(
    handle, device, num_seqs, num_heads, num_kv_heads, head_size, 
    block_size, max_step_len, dtype=InfiniDtype.F16, sync=None,
):
    print(f"Testing Multi-turn PagedAttention | num_seqs={num_seqs}, dtype={InfiniDtypeNames[dtype]}")

    num_blocks = 8192
    manager = SimpleCacheManager(num_blocks, block_size)
    scale = head_size ** -0.5
    
    # 初始化持久化物理缓存 (模拟显存池)
    k_cache = TestTensor((num_blocks, num_kv_heads, block_size, head_size), None, dtype, device)
    v_cache = TestTensor((num_blocks, num_kv_heads, block_size, head_size), None, dtype, device)
    
    num_rounds = 2
    for r in range(num_rounds):
        print(f"--- Round {r+1} ---")
        
        # 1. 模拟调度与物理写入
        new_lens_torch = torch.randint(1, max_step_len + 1, (num_seqs,), dtype=torch.int32)
        total_lens_list = []
        all_block_tables = []
        
        # 准备本轮输入的 Q (带 Padding 形状以配合 C 接口)
        max_new_len = new_lens_torch.max().item()
        q_new_torch = torch.zeros(num_seqs, max_new_len, num_heads, head_size)
        
        for i in range(num_seqs):
            cur_new_len = new_lens_torch[i].item()
            table, total_len = manager.allocate_slots(i, cur_new_len)
            total_lens_list.append(total_len)
            all_block_tables.append(table)
            
            # 生成新 KV 并写入持久化物理缓存 (模拟 PagedCaching 行为)
            k_new = torch.randn(cur_new_len, num_kv_heads, head_size)
            v_new = torch.randn(cur_new_len, num_kv_heads, head_size)
            q_val = torch.randn(cur_new_len, num_heads, head_size)
            q_new_torch[i, :cur_new_len, :, :] = q_val
            
            history_len = total_len - cur_new_len
            for t in range(cur_new_len):
                logical_pos = history_len + t
                b_id = table[logical_pos // block_size]
                off = logical_pos % block_size
                k_cache.torch_tensor()[b_id, :, off, :] = k_new[t]
                v_cache.torch_tensor()[b_id, :, off, :] = v_new[t]

        # 2. 准备算子 Tensor
        q_new = TestTensor.from_torch(q_new_torch, dtype, device)
        out = TestTensor((num_seqs, max_new_len, num_heads, head_size), None, dtype, device)
        seq_lens = TestTensor.from_torch(torch.tensor(total_lens_list, dtype=torch.int32), InfiniDtype.I32, device)
        
        max_blocks = max(len(t) for t in all_block_tables)
        padded_tables = [t + [0]*(max_blocks - len(t)) for t in all_block_tables]
        block_tables = TestTensor.from_torch(torch.tensor(padded_tables, dtype=torch.int32), InfiniDtype.I32, device)

        # 3. Reference 计算
        ans = ref_paged_attention_multi_turn(
            q_new.torch_tensor(), k_cache.torch_tensor(), v_cache.torch_tensor(),
            block_tables.torch_tensor(), seq_lens.torch_tensor(), new_lens_torch, scale
        )

        # ======================================================================
        # 4. 执行算子 (完善后的 C++ 接口调用)
        # ======================================================================
        
        # 将本轮新增长度 new_lens_torch 转换为 TestTensor 以便传递给 C++
        new_lens = TestTensor.from_torch(new_lens_torch, InfiniDtype.I32, device)

        descriptor = infiniopOperatorDescriptor_t()
        
        # 创建描述符：注意参数顺序，增加了 new_lens.descriptor
        check_error(LIBINFINIOP.infiniopCreatePagedAttentionPrefillDescriptor(
            handle, 
            ctypes.byref(descriptor),
            out.descriptor, 
            q_new.descriptor,
            k_cache.descriptor, 
            v_cache.descriptor,
            block_tables.descriptor, 
            seq_lens.descriptor,
            new_lens.descriptor,    # <-- 对齐底层实现中的 new_lens_desc
            None,                   # alibi_slopes_desc (传入空)
            scale
        ))

        # 获取并准备 Workspace
        workspace_size = c_uint64(0)
        check_error(LIBINFINIOP.infiniopGetPagedAttentionPrefillWorkspaceSize(
            descriptor, 
            ctypes.byref(workspace_size)
        ))
        workspace = TestWorkspace(workspace_size.value, device)

        # 执行 Prefill 计算：注意增加了 new_lens.data()
        check_error(LIBINFINIOP.infiniopPagedAttentionPrefill(
            descriptor, 
            workspace.data(), 
            workspace_size.value,
            out.data(), 
            q_new.data(),
            k_cache.data(), 
            v_cache.data(),
            block_tables.data(), 
            seq_lens.data(),
            new_lens.data(),        # <-- 对齐底层实现中的 new_lens 指针
            None,                   # alibi_slopes
            None                    # stream (通常传入空，由内部自动获取当前流)
        ))

        if sync: sync()

        # ======================================================================
        # 5. 验证
        # ======================================================================
        atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
        # compare out.actual_tensor() with reference result ans
        assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)
        
        # 清理
        check_error(LIBINFINIOP.infiniopDestroyPagedAttentionPrefillDescriptor(descriptor))
        print(f"Round {r+1} verified. Seq lens: {total_lens_list}")

# ==============================================================================
# 配置与启动
# ==============================================================================
_TEST_CASES_ = [
    # (num_seqs, num_heads, num_kv_heads, head_size, block_size, max_step_len)
    (2, 8, 8, 128, 16, 32),
    # (4, 16, 16, 64, 8, 64),
]

_TENSOR_DTYPES = [InfiniDtype.F16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
}

if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)
    print("\033[92mMulti-turn PagedAttention test passed!\033[0m")
