# GLM-5.2-W4A8 InfiniCore 算子接入进度

记录环境：TG150 .58 `/home/wuwei/InfiniCore`，容器 `pepe`，Iluvatar BI-V150 (`ivcore11`)。

## 已接入/验证

| Trace 算子类别 | 当前 InfiniCore 接入状态 | 验证 |
|---|---|---|
| `pyinfer::perf::dynamic_scaled_int8_quant` | 新增 `infinicore.dynamic_scaled_int8_quant`，通过 `vllm_iluvatar/_C` dlopen+dlsym 桥接 | F16/BF16 torch zero-copy 数值验证通过，scale/output 与参考一致 |
| `pyinfer::perf::fused_add_rms_norm` | 新增 `infinicore.add_rms_norm_inplace`，通过 `vllm_iluvatar/_C` 桥接 | F16/BF16 数值验证通过 |
| `silu_and_mul` | 新增/启用 InfiniOp Iluvatar 路径，复用 NVIDIA CUDA kernel 结构 | `python3 test/infiniop/silu_and_mul.py --iluvatar` 通过 |
| `moe_sum` | InfiniOp NVIDIA backend 增加 Iluvatar dispatch/编译 guard | `deepseek_moe.py --iluvatar` 覆盖通过 |
| `pyinfer::perf::topk_softmax` / `topk_sigmoid` | 新增 `infinicore.moe_topk_softmax_vllm` / `moe_topk_sigmoid_vllm`，通过 `vllm_iluvatar/_C` 桥接，三输出匹配 vLLM trace: weights、expert ids、source rows | F16/BF16/F32、renormalize true/false、correction_bias 验证通过 |
| `pyinfer::perf::moe_grouped_topk` / `_moe_C.grouped_topk` | 新增 `infinicore.grouped_topk_vllm`，通过 `vllm_iluvatar/_C` 桥接；仅开放 correction-bias 路径，no-bias 在 Python/C++ 层显式拒绝 | 与 vLLM 实际生产入口 `torch.ops._moe_C.grouped_topk` 在 F16/BF16、softmax/sigmoid、renormalize true/false 下逐位一致；F16 也匹配 `vllm_iluvatar` torch reference；BF16 与 torch reference 存在偏差，记录为复用生产 perf 语义的风险 |
| `pyinfer::cuinfer::scaled_mm_w4a8` / `_C.scaled_mm_w4a8` | 新增 `infinicore.scaled_mm_w4a8`，通过 `vllm_iluvatar/_C` 的 cuinfer 符号桥接；用于 dense W4A8 linear packed GEMM，当前不改权重布局 | 与 vLLM 实际生产入口 `torch.ops._C.scaled_mm_w4a8` 在 F16/BF16、有无 bias 下逐位一致 |
| `pyinfer::cuinfer::w4a8_group_gemm` / `_C.w4a8_group_gemm` | 新增 `infinicore.w4a8_group_gemm_`，通过 `vllm_iluvatar/_C` 的 cuinfer 符号桥接；用于 MoE W4A8 group GEMM | 与 vLLM 实际生产入口 `torch.ops._C.w4a8_group_gemm` 在 F16/BF16、有无 bias、有无 sorted_token_ids、prefill/decode 标志下逐位一致；未声称匹配 torch fallback，因为 fallback 对构造样本与 production 不一致 |
| `pyinfer::cuinfer::w8a8_group_gemm` / `_C.w8a8_group_gemm` | 新增 `infinicore.w8a8_group_gemm_`，通过 `vllm_iluvatar/_C` 的 cuinfer 符号桥接；仅开放 `trans_weight=True` 的 TN prefill 路径 | 与 vLLM 实际生产入口 `torch.ops._C.w8a8_group_gemm` 在 F16/BF16、有无 bias、有无 sorted_token_ids、多个 E/M/K/N shape 下逐位一致；`is_decode=True` 会触发 vLLM/CUINFER internal error，已在 Python/C++ 层显式 guard 禁用 |
| `moe_topk_softmax` | InfiniOp NVIDIA backend 增加 Iluvatar dispatch/编译 guard | `topksoftmax.py --iluvatar`、`deepseek_moe.py --iluvatar` 通过 |
| `moe_topk_sigmoid` | InfiniOp NVIDIA backend 增加 Iluvatar dispatch/编译 guard | `deepseek_moe.py --iluvatar` 覆盖通过 |
| `rms_norm` / `rope` / `mrope` / `kv_caching` / `topksoftmax` | 仓库已有 InfiniCore/InfiniOp 接口和 Iluvatar 可运行路径 | 既有测试曾通过；本轮重跑了 `topksoftmax` |
| `pyinfer::perf::rotary_embedding` | 暂不 bridge；继续使用已验证的 InfiniCore `rope/mrope` 路径 | 直接调用 vLLM perf 对 F16/BF16、2D/3D、Neox/GPT-J 均产生 NaN；已撤回 bridge，避免暴露错误接口 |

## 已明确不接入的错误路径

- `scaled_mm_int8` 的 NVIDIA path 曾尝试打开 Iluvatar dispatch，但首个 BF16 case 数值失败；已撤回，当前 `scaled_mm` 下没有残留 `ENABLE_ILUVATAR` / `INFINI_DEVICE_ILUVATAR` 修改。
- GLM 的 W4A8/GEMM 热路径应继续优先复用 cuinfer/vLLM 已用 so，而不是暴露未验证的 InfiniOp int8 GEMM 路径。

## 当前新增接口

- C++ header: `include/infinicore/ops/dynamic_scaled_int8_quant.hpp`
- Python: `infinicore.dynamic_scaled_int8_quant(input, input_scales, out=None)`
  - `input`: F16/BF16 contiguous，last dim 为 hidden size
  - `input_scales`: F32 contiguous，numel = `input.numel / input.shape[-1]`
  - `out`: optional I8 contiguous，shape 与 input 相同
- C++/Python: `add_rms_norm_inplace(input, residual, weight, epsilon=1e-5)`
- C++/Python: `concat_mla_q(ql_nope, q_pe, out=None)`，当前限制 GLM MLA 维度 `512 + 64 -> 576`
- C++/Python: `concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype="auto", scale=...)`
- C++/Python: `concat_and_cache_mla_int8(kv_c_int8, kv_c_scale, k_pe_int8, k_pe_scale, kv_cache, kv_cache_scale, slot_mapping)`
- C++/Python: `moe_topk_softmax_vllm(gating_output, topk, renormalize=False, correction_bias=None, out=None)`
- C++/Python: `moe_topk_sigmoid_vllm(gating_output, topk, renormalize=False, correction_bias=None, out=None)`
- C++/Python: `grouped_topk_vllm(scores, num_expert_group, topk_group, topk, renormalize, routed_scaling_factor=1.0, bias=..., scoring_func="softmax", out=None)`
  - 当前必须传 `bias` / correction bias；no-bias 路径因 vLLM perf 与 reference 不一致而禁用。
  - 对 torch tensor 做 zero-copy 验证时使用 `infinicore.tensor.from_torch(...)` 包装输入和 `out` buffer。
- C++/Python: `scaled_mm_w4a8(a, b, a_scales, b_scales, bias=None, trans_weight=False, out=None)`
  - `a`: int8 activation `(M,K)`；`b`: packed int4-as-int8 weight `(K,N/2)` when `trans_weight=False`；`a_scales`: F32 `(M,1)`；`b_scales`: F32 `(N,1)`；`out`: F16/BF16 `(M,N)`。
  - 当前验证的是 dense W4A8 NN layout，与 vLLM `scaled_mm_w4a8_packed_fwd`/`torch.ops._C.scaled_mm_w4a8` 路径一致。
- C++/Python: `w4a8_group_gemm_(out, input, weight, input_scale, weight_scale, tokens_per_experts, sorted_token_ids=None, bias=None, trans_weight=True, is_decode=False)`
  - `input`: int8 `(M,K)`；`weight`: packed int4-as-int8 `(E,N,K/2)` for TN；`input_scale`: F32 `(M,1)`；`weight_scale`: F32 `(E,N,1)`；`tokens_per_experts`: int32 `(E,)`；`out`: F16/BF16 `(M,N)`。
  - 与 vLLM production op 逐位一致；prefill 构造样本中 `tokens_per_experts` 使用 CPU int32，decode 构造样本中使用 GPU int32，均已覆盖。
- C++/Python: `w8a8_group_gemm_(out, input, weight, input_scale, weight_scale, tokens_per_experts, sorted_token_ids=None, bias=None, trans_weight=True, is_decode=False)`
  - `input`: int8 `(M,K)`；`weight`: int8 `(E,N,K)` for TN；`input_scale`: F32 `(M,1)`；`weight_scale`: F32 `(E,N,1)`；`tokens_per_experts`: int32 `(E,)`；`out`: F16/BF16 `(M,N)`。
  - 当前只开放 `trans_weight=True` 且 `is_decode=False`。`trans_weight=False` 与 `is_decode=True` 均会在 Python/C++ 层报错；后者是因为 vLLM/CUINFER production kernel decode 分支在当前环境触发 `CUINFER_STATUS_INTERNAL_ERROR`。
  - 与 vLLM production prefill op 逐位一致；验证覆盖 F16/BF16、有无 bias、有无 sorted_token_ids，以及 `(E,M,K,N)=(4,16,128,256),(4,64,256,512),(8,128,256,512)`。

## 待接入/待确认

| Trace 符号 | 建议路径 | 备注 |
|---|---|---|
| `concat_and_cache_mla_rope_fused` | 暂不接入 | 直接对比 vLLM perf 与其 torch fallback：q_pe/k_pe 原地 RoPE 结果不一致，cache 近似一致；为避免暴露错误语义，已撤回 bridge |
| ixCCL send/recv/allreduce/allgather | 归入 distributed/ccl 层 | 不属于单 GPU kernel；需单独检查 InfiniCore CCL 接口 |

## 本轮验证命令摘要

```bash
xmake f --iluvatar-gpu=true --ccl=false --aten=true --iluvatar_arch=ivcore11 --cuda=/usr/local/corex-4.5.0.20260619 -cv
xmake build infinicore_cpp_api
xmake build _infinicore
xmake install infinicore_cpp_api
xmake install _infinicore
xmake build infiniop-iluvatar
xmake build infiniop
xmake install infiniop
python3 test/infiniop/silu_and_mul.py --iluvatar
python3 test/infiniop/topksoftmax.py --iluvatar
python3 test/infiniop/deepseek_moe.py --iluvatar
# grouped_topk_vllm: Python zero-copy torch buffer 测试，对比 torch.ops._moe_C.grouped_topk，并验证 no-bias guard
# scaled_mm_w4a8: Python zero-copy torch buffer 测试，对比 torch.ops._C.scaled_mm_w4a8，F16/BF16、有无 bias 逐位一致
# w4a8_group_gemm_: Python zero-copy torch buffer 测试，对比 torch.ops._C.w4a8_group_gemm，F16/BF16、有无 bias、有无 sorted_token_ids、prefill/decode 逐位一致
# w8a8_group_gemm_: Python zero-copy torch buffer 测试，对比 torch.ops._C.w8a8_group_gemm，F16/BF16、有无 bias、有无 sorted_token_ids、prefill 逐位一致；decode/trans_weight guard 已验证
```
