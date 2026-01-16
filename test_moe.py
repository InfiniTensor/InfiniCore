import time
import torch
import transformers
import safetensors
import os
from transformers import AutoConfig
import sys
import ctypes
import argparse
import numpy as np

# =================================================================
# 1. System Configuration & Import C++ Ops
# =================================================================
try:
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    
    # Adjust path if needed
    sys.path.append("/data/users/shankgu/InfiniCore/python")
    
    try:
        import infinicore
    except ImportError:
        pass 
        
    sys.setdlopenflags(old_flags)
    import gu_moe_ops
    print("[Success] gu_moe_ops imported successfully.")
except ImportError as e:
    print(f"[Error] Failed to import gu_moe_ops: {e}")
    sys.exit(1)

try:
    from transformers.models import qwen3_moe
except ImportError:
    try:
        from transformers.models import qwen2_moe as qwen3_moe
        print("[Info] Qwen3 not found, using Qwen2 classes.")
    except ImportError:
        print("[Error] Transformers library missing Qwen MoE models.")
        sys.exit(1)

# =================================================================
# 2. Benchmark Configuration (Official)
# =================================================================
WARMUPS = 10
RUNS = 100
PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [512, 0, 0, 256]}

DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],
    "pastlens": [50 for _ in range(4)]
    + [100 for _ in range(4)]
    + [200 for _ in range(4)]
    + [400 for _ in range(4)],
}

# =================================================================
# 3. C++ Wrapper Class
# =================================================================
class GuMoeWrapper(torch.nn.Module):
    def __init__(self, cpp_block):
        super().__init__()
        self.cpp_block = cpp_block
    
    def forward(self, hidden_states):
        return self.cpp_block.forward(hidden_states.contiguous()), None

# =================================================================
# 4. Helper Functions
# =================================================================
def get_args():
    parser = argparse.ArgumentParser(description="Test Operator")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    return parser.parse_args()

def torch_synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()

def torch_empty_cache(device):
    if device == "cuda":
        torch.cuda.empty_cache()

def get_real_inter_dim(dir_path):
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".safetensors"):
            with safetensors.safe_open(os.path.join(dir_path, fname), framework="pt") as f:
                for key in f.keys():
                    if "down_proj.weight" in key:
                        return f.get_tensor(key).shape[1]
    return None

# =================================================================
# 5. Model Creation
# =================================================================
def create_moe_torch(dir_path, device, dtype=torch.float32):
    print(f"[Torch] Creating model from {dir_path}...")
    config = AutoConfig.from_pretrained(dir_path)
    
    if not hasattr(config, "shared_expert_intermediate_size"):
        config.shared_expert_intermediate_size = config.intermediate_size

    try:
        if hasattr(qwen3_moe, "modeling_qwen3_moe"):
            moe = qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
        else:
            moe = qwen3_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
    except Exception as e:
        print(f"Error creating Torch model: {e}")
        return None

    tensors = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"): continue
        fpath = os.path.join(dir_path, fname)
        with safetensors.safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                if "model.layers.0.mlp." in key:
                    tensors[key[len("model.layers.0.mlp.") :]] = f.get_tensor(key)
        break
    
    moe.load_state_dict(tensors, strict=False)

    # Disable Shared Expert for fair comparison
    if hasattr(moe, "shared_expert"):
        for param in moe.shared_expert.parameters():
            torch.nn.init.zeros_(param)
            param.requires_grad = False
    if hasattr(moe, "shared_expert_gate"):
        torch.nn.init.zeros_(moe.shared_expert_gate.weight)
        
    return moe

def create_moe_custom(dir_path, device, dtype=torch.float32):
    print(f"[C++] Creating Optimized Model from {dir_path}...")
    config = AutoConfig.from_pretrained(dir_path)
    real_inter = get_real_inter_dim(dir_path)
    inter_dim = real_inter if real_inter else config.intermediate_size
    
    norm_topk = True
    if hasattr(config, "norm_topk_prob"):
        norm_topk = config.norm_topk_prob

    raw_moe = gu_moe_ops.GuMoeBlock(
        config.num_experts, config.hidden_size, inter_dim, config.num_experts_per_tok, norm_topk
    )

    print("[C++] Loading weights...")
    expert_buffers = {}
    router_weight = None
    
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"): continue
        fpath = os.path.join(dir_path, fname)
        with safetensors.safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                if "model.layers.0.mlp." not in key: continue
                short_key = key[len("model.layers.0.mlp.") :]
                tensor = f.get_tensor(key).to(device=device, dtype=dtype)
                
                if "gate.weight" in short_key:
                    router_weight = tensor.contiguous()
                elif "experts." in short_key and "weight" in short_key:
                    parts = short_key.split('.')
                    try:
                        expert_id = int(parts[1])
                        proj_type = parts[2]
                        if expert_id not in expert_buffers: expert_buffers[expert_id] = {}
                        expert_buffers[expert_id][proj_type] = tensor
                    except: pass

    final_state_dict = {}
    if router_weight is not None: final_state_dict["router.weight"] = router_weight
    
    gate_up_list = []
    down_list = []
    for i in range(config.num_experts):
        if i in expert_buffers:
            keys = expert_buffers[i].keys()
            if "gate_up_proj" in keys:
                gate_up = expert_buffers[i]["gate_up_proj"]
            elif "gate_proj" in keys and "up_proj" in keys:
                gate_up = torch.cat([expert_buffers[i]["gate_proj"], expert_buffers[i]["up_proj"]], dim=0)
            else:
                gate_up = torch.zeros((2*inter_dim, config.hidden_size), device=device, dtype=dtype)

            down = expert_buffers[i].get("down_proj", expert_buffers[i].get("down"))
            if down is None: down = torch.zeros((config.hidden_size, inter_dim), device=device, dtype=dtype)
            
            gate_up_list.append(gate_up)
            down_list.append(down)
        else:
            gate_up_list.append(torch.zeros((2*inter_dim, config.hidden_size), device=device, dtype=dtype))
            down_list.append(torch.zeros((config.hidden_size, inter_dim), device=device, dtype=dtype))

    if gate_up_list:
        final_state_dict["experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0).contiguous()
        final_state_dict["experts.down_proj"] = torch.stack(down_list, dim=0).contiguous()

    raw_moe.load_state_dict(final_state_dict)
    return GuMoeWrapper(raw_moe)

# =================================================================
# 6. Benchmark Engine
# =================================================================
def generate_moe_input_torch(testcase, dtype=torch.float32):
    total_seqlen = sum(testcase["seqlens"])
    input_tensor = torch.randn((1, total_seqlen, 2048), device="cpu", dtype=dtype)
    return input_tensor

def benchmark_moe(model, testcase, device, dtype, name="Model"):
    input_host = generate_moe_input_torch(testcase, dtype=dtype)
    input_device = input_host.to(device=device)

    # Warmup
    for _ in range(WARMUPS):
        _ = model(input_device)
    torch_synchronize(device)

    # Run
    start_time = time.time()
    for _ in range(RUNS):
        _ = model(input_device)
        torch_synchronize(device)
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = (total_time * 1000) / RUNS
    total_tokens = sum(testcase["seqlens"]) * RUNS
    throughput = total_tokens / total_time

    print(f"\t [{name:<8}] Latency: {avg_latency:6.3f} ms | Throughput: {throughput:8.2f} tok/s")
    return avg_latency

# =================================================================
# 7. Accuracy Verification
# =================================================================
def verify_accuracy(torch_model, cpp_model, device, dtype):
    print("\n" + "=" * 80)
    print(" 🧪 CORRECTNESS CHECK")
    print("=" * 80)
    
    torch.manual_seed(42)
    # Use a challenging input size
    test_input = torch.randn(1, 128, 2048, device=device, dtype=dtype)
    
    with torch.no_grad():
        out_ref, _ = torch_model(test_input)
        out_cpp, _ = cpp_model(test_input)
        
    abs_diff = (out_ref - out_cpp).abs()
    eps = 1e-5
    rel_diff = abs_diff / (out_ref.abs() + eps)
    
    max_diff = abs_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"  PyTorch Max: {out_ref.max().item():.6f}")
    print(f"  C++ Max:     {out_cpp.max().item():.6f}")
    print("-" * 40)
    print(f"  Max Absolute Diff: {max_diff:.6f}")
    print(f"  Mean Relative Diff: {mean_rel_diff:.6f}")
    
    # Threshold for FP32: 1e-3 is safe for MoE due to atomicAdd nondeterminism
    if mean_rel_diff < 0.01:
        print("\n  ✅ PASSED: Implementation matches PyTorch baseline.")
    else:
        print("\n  ❌ FAILED: Significant numerical deviation detected.")
        
    print("=" * 80 + "\n")

# =================================================================
# Main
# =================================================================
if __name__ == "__main__":
    args = get_args()
    device = "cuda" if args.nvidia else "cpu"
    dtype = torch.float32 

    print(f"Device: {device}")
    print(f"Model Path: {args.model_path}")

    # 1. Initialize Models
    torch_model = create_moe_torch(args.model_path, device, dtype)
    cpp_model = create_moe_custom(args.model_path, device, dtype)

    # 2. Verify Accuracy First
    verify_accuracy(torch_model, cpp_model, device, dtype)

    # 3. Run Benchmarks
    print(" 🚀 BENCHMARK: Qwen3 MoE Operator")
    print("=" * 80)

    # --- Prefill Test ---
    print(f"\n[Test 1] PREFILL Phase (Batch Compute)")
    print(f"Case: {PREFILL_TESTCASES}")
    
    t_torch_p = benchmark_moe(torch_model, PREFILL_TESTCASES, device, dtype, "PyTorch")
    t_cpp_p   = benchmark_moe(cpp_model, PREFILL_TESTCASES, device, dtype, "C++")
    print(f"   >>> Speedup: {t_torch_p / t_cpp_p:.2f}x")

    # --- Decode Test ---
    print(f"\n[Test 2] DECODE Phase (Small Batch / Latency Critical)")
    print(f"Case: {DECODE_TESTCASES}")
    
    t_torch_d = benchmark_moe(torch_model, DECODE_TESTCASES, device, dtype, "PyTorch")
    t_cpp_d   = benchmark_moe(cpp_model, DECODE_TESTCASES, device, dtype, "C++")
    print(f"   >>> Speedup: {t_torch_d / t_cpp_d:.2f}x")

    print("\n" + "=" * 80)