import torch
import ctypes
from ctypes import c_uint64
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
# 模拟上层调度器的逻辑
# ==============================================================================
class SimpleCacheManager:
    """模拟管理物理块分配的调度器"""
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        # request_id -> list of physical_block_ids
        self.request_to_blocks = {}
        # request_id -> current_length
        self.request_to_len = {}

    def allocate_slots(self, request_id, num_new_tokens):
        if request_id not in self.request_to_len:
            self.request_to_len[request_id] = 0
            self.request_to_blocks[request_id] = []
        
        start_pos = self.request_to_len[request_id]
        new_total_len = start_pos + num_new_tokens
        
        # 计算总共需要多少块
        needed_blocks = (new_total_len + self.block_size - 1) // self.block_size
        added_blocks = needed_blocks - len(self.request_to_blocks[request_id])
        
        # 分配新块
        for _ in range(added_blocks):
            self.request_to_blocks[request_id].append(self.free_blocks.pop(0))
            
        # 计算物理 slot_mapping
        slots = []
        for i in range(start_pos, new_total_len):
            block_idx_in_seq = i // self.block_size
            block_offset = i % self.block_size
            physical_block_id = self.request_to_blocks[request_id][block_idx_in_seq]
            slots.append(physical_block_id * self.block_size + block_offset)
            
        self.request_to_len[request_id] = new_total_len
        return torch.tensor(slots, dtype=torch.int32)

# ==============================================================================
# Reference Implementation (支持增量写入)
# ==============================================================================
def ref_paged_caching_incremental(key_new, value_new, key_cache_pool, value_cache_pool, slot_mapping_new):
    """
    参考实现：仅将新的 key/value 写入对应的物理槽位，不改变 pool 中其他位置
    """
    ntok_new = key_new.shape[0]
    block_size = key_cache_pool.shape[2]

    # 直接在传入的 pool 上模拟写入（测试代码中会先 clone）
    for i in range(ntok_new):
        slot = slot_mapping_new[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size

        key_cache_pool[block_idx, :, block_offset, :] = key_new[i]
        value_cache_pool[block_idx, :, block_offset, :] = value_new[i]

    return key_cache_pool, value_cache_pool


# ==============================================================================
# Test Operator 实现
# ==============================================================================
def test(
    handle,
    device,
    num_seqs,       # 并发的 Request 数量
    max_step_len,   # 每轮新增的 Token 长度上限
    num_kv_heads,
    head_size,
    block_size,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(f"Testing Multi-turn PagedCaching on {InfiniDeviceNames[device]} | num_seqs={num_seqs}")

    num_blocks = 8192
    manager = SimpleCacheManager(num_blocks, block_size)
    
    # 模拟两轮对话
    num_rounds = 3
    
    # 初始化全局 Cache Pool
    k_cache_pool = TestTensor((num_blocks, num_kv_heads, block_size, head_size), None, dtype, device)
    v_cache_pool = TestTensor((num_blocks, num_kv_heads, block_size, head_size), None, dtype, device)
    
    # 克隆用于 Reference 的 Pool
    k_cache_ref = k_cache_pool.torch_tensor().clone()
    v_cache_ref = v_cache_pool.torch_tensor().clone()

    for r in range(num_rounds):
        print(f"--- Round {r+1} ---")
        
        # 1. 模拟上层调度：为每个请求分配本轮新增的 Token 数量和物理槽位
        round_ntok_list = torch.randint(1, max_step_len + 1, (num_seqs,), dtype=torch.int32)
        all_slots_list = []
        all_k_new_list = []
        all_v_new_list = []
        
        for i in range(num_seqs):
            n_new = round_ntok_list[i].item()
            # 获取物理槽位
            slots = manager.allocate_slots(request_id=i, num_new_tokens=n_new)
            all_slots_list.append(slots)
            
            # 生成该 Request 本轮新增的 KV 数据
            all_k_new_list.append(torch.randn(n_new, num_kv_heads, head_size))
            all_v_new_list.append(torch.randn(n_new, num_kv_heads, head_size))
            
        # 拼接成算子输入形式 (ntok_total_this_round, nkvh, dh)
        k_new_torch = torch.cat(all_k_new_list, dim=0)
        v_new_torch = torch.cat(all_v_new_list, dim=0)
        slot_mapping_torch = torch.cat(all_slots_list, dim=0)
        
        # 创建本轮的测试张量
        k_in = TestTensor.from_torch(k_new_torch, dtype, device)
        v_in = TestTensor.from_torch(v_new_torch, dtype, device)
        slot_mapping = TestTensor.from_torch(slot_mapping_torch, InfiniDtype.I32, device)

        # 2. Reference 计算
        k_cache_ref, v_cache_ref = ref_paged_caching_incremental(
            k_in.torch_tensor(), v_in.torch_tensor(), 
            k_cache_ref, v_cache_ref, 
            slot_mapping.torch_tensor()
        )

        # 3. 算子执行
        descriptor = infiniopOperatorDescriptor_t()
        check_error(LIBINFINIOP.infiniopCreatePagedCachingDescriptor(
            handle, ctypes.byref(descriptor),
            k_in.descriptor, v_in.descriptor,
            k_cache_pool.descriptor, v_cache_pool.descriptor,
            slot_mapping.descriptor
        ))

        workspace_size = c_uint64(0)
        check_error(LIBINFINIOP.infiniopGetPagedCachingWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
        workspace = TestWorkspace(workspace_size.value, device)

        # 执行
        check_error(LIBINFINIOP.infiniopPagedCaching(
            descriptor, workspace.data(), workspace_size.value,
            k_in.data(), v_in.data(),
            k_cache_pool.data(), v_cache_pool.data(),
            slot_mapping.data(), None
        ))

        if sync: sync()

        # 4. 验证本轮写入后的 Cache 状态
        atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
        assert torch.allclose(k_cache_pool.actual_tensor(), k_cache_ref, atol=atol, rtol=rtol)
        assert torch.allclose(v_cache_pool.actual_tensor(), v_cache_ref, atol=atol, rtol=rtol)
        
        # 清理本轮描述符
        check_error(LIBINFINIOP.infiniopDestroyPagedCachingDescriptor(descriptor))
        print(f"Round {r+1} verified.")

# ==============================================================================
# 配置与启动
# ==============================================================================
_TEST_CASES_ = [
    # (num_seqs, max_step_len, num_kv_heads, head_size, block_size)
    (2, 64, 8, 128, 16),
    (8, 128, 32, 128, 16),
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
    print("\033[92mMulti-turn PagedCaching test passed!\033[0m")
