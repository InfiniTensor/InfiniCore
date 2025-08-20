from ast import List
import numpy as np
import gguf
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16

from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def create_attention_with_weights(config_dict, weights):
    """创建attention模块并设置权重"""
    config = DeepseekV3Config(** config_dict)
    attention = DeepseekV3Attention(config, layer_idx=0)
    attention.eval()
    
    # 将随机生成的权重设置到模型中
    if weights['w_q_a'] is not None:
        # Q的LoRA投影
        attention.q_a_proj.weight.data = torch.from_numpy(weights['w_q_a'].T.astype(np.float64))
        if weights['bias_q_a'] is not None:
            attention.q_a_proj.bias.data = torch.from_numpy(weights['bias_q_a'].astype(np.float64))
        attention.q_a_layernorm.weight.data = torch.from_numpy(weights['w_q_a_norm'].astype(np.float64))
        attention.q_b_proj.weight.data = torch.from_numpy(weights['w_q_b'].T.astype(np.float64))
    else:
        # 无LoRA的Q投影
        attention.q_proj.weight.data = torch.from_numpy(weights['w_q_b'].T.astype(np.float64))
    
    # KV的投影（注意：KV头数与Q不同）
    attention.kv_a_proj_with_mqa.weight.data = torch.from_numpy(weights['w_kv_a'].T.astype(np.float64))
    if weights['bias_kv_a'] is not None:
        attention.kv_a_proj_with_mqa.bias.data = torch.from_numpy(weights['bias_kv_a'].astype(np.float64))
    attention.kv_a_layernorm.weight.data = torch.from_numpy(weights['w_kv_a_norm'].astype(np.float64))
    attention.kv_b_proj.weight.data = torch.from_numpy(weights['w_kv_b'].T.astype(np.float64))
    
    # 输出投影
    attention.o_proj.weight.data = torch.from_numpy(weights['w_o'].T.astype(np.float64))
    if weights['bias_o'] is not None:
        attention.o_proj.bias.data = torch.from_numpy(weights['bias_o'].astype(np.float64))
    
    return attention


class MultiHeadLatentAttentionTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        config_dict: dict,
        input_strides: List[int] | None,
        output_strides: List[int] | None, 
        weight_strides: List[int] | None,
        has_bias: bool = True,
        use_q_lora: bool = True,
        dtype: np.dtype = np.float32,
        seed: int = 42,
    ):
        super().__init__("multi_head_latent_attention")
        
        self.input_shape = input_shape
        self.config_dict = config_dict.copy()
        self.config_dict['attention_bias'] = has_bias
        if not use_q_lora:
            self.config_dict['q_lora_rank'] = None
        
        self.has_bias = has_bias
        self.use_q_lora = use_q_lora
        self.input_strides = input_strides
        self.output_strides = output_strides
        self.weight_strides = weight_strides or {}
        self.dtype = dtype
        self.seed = seed
        
        # 计算输出形状 (batch_size, seq_len, hidden_size)
        self.output_shape = input_shape[:-1] + (config_dict['hidden_size'],)
        
        # 验证配置
        self._validate_config()
        
        # 生成测试数据
        self._generate_test_data()

    def _validate_config(self):
        """验证配置的合理性（补充关键维度检查）"""
        # 基础维度检查
        assert self.config_dict['qk_head_dim'] == self.config_dict['qk_nope_head_dim'] + self.config_dict['qk_rope_head_dim'], \
            f"qk_head_dim({self.config_dict['qk_head_dim']}) should equal qk_nope_head_dim + qk_rope_head_dim"
        
        # Grouped-Query Attention检查
        assert self.config_dict['num_attention_heads'] % self.config_dict['num_key_value_heads'] == 0, \
            "num_attention_heads should be divisible by num_key_value_heads"
        self.num_key_value_groups = self.config_dict['num_attention_heads'] // self.config_dict['num_key_value_heads']
        
        # 输入维度检查
        assert self.config_dict['hidden_size'] == self.input_shape[-1], \
            f"hidden_size({self.config_dict['hidden_size']}) should match input last dimension"
        
        # RoPE维度检查
        assert self.config_dict['qk_rope_head_dim'] > 0, "qk_rope_head_dim must be positive"

    def _generate_test_data(self):
        """生成测试数据（输入、权重）"""
        np.random.seed(self.seed)
        
        # 生成输入张量
        if self.input_strides is not None:
            # 处理自定义步长（示例实现）
            self.input = np.random.randn(*self.input_shape).astype(self.dtype) * 0.1
            self.input = np.lib.stride_tricks.as_strided(
                self.input, 
                shape=self.input_shape,
                strides=tuple(s * self.input.itemsize for s in self.input_strides)
            )
        else:
            self.input = np.random.randn(*self.input_shape).astype(self.dtype) * 0.1
        
        # 生成权重
        self.weights = self._generate_weights()
        
        # 初始化输出张量
        if self.output_strides is not None:
            self.output = np.empty(self.output_shape, dtype=self.dtype)
            self.output = np.lib.stride_tricks.as_strided(
                self.output,
                shape=self.output_shape,
                strides=tuple(s * self.output.itemsize for s in self.output_strides)
            )
        else:
            self.output = np.empty(self.output_shape, dtype=self.dtype)

    def _generate_weights(self):
        """生成权重（修正KV头数对应的维度）"""
        hidden_size = self.config_dict['hidden_size']
        num_q_heads = self.config_dict['num_attention_heads']
        num_kv_heads = self.config_dict['num_key_value_heads']  # 关键修正：使用KV头数而非Q头数
        q_lora_rank = self.config_dict.get('q_lora_rank')
        kv_lora_rank = self.config_dict['kv_lora_rank']
        qk_head_dim = self.config_dict['qk_head_dim']
        qk_nope_head_dim = self.config_dict['qk_nope_head_dim']
        qk_rope_head_dim = self.config_dict['qk_rope_head_dim']
        v_head_dim = self.config_dict['v_head_dim']
        
        # 权重初始化缩放因子
        weight_scale = 1.0 / np.sqrt(hidden_size)
        weights = {}
        
        # Q投影权重（LoRA或直接投影）
        if q_lora_rank is not None:
            weights['w_q_a'] = np.random.randn(hidden_size, q_lora_rank).astype(self.dtype) * weight_scale
            weights['bias_q_a'] = np.zeros(q_lora_rank).astype(self.dtype) if self.has_bias else None
            weights['w_q_a_norm'] = np.ones(q_lora_rank).astype(self.dtype)  # RMSNorm权重
            weights['w_q_b'] = np.random.randn(q_lora_rank, num_q_heads * qk_head_dim).astype(self.dtype) * (1.0 / np.sqrt(q_lora_rank))
        else:
            weights['w_q_a'] = None
            weights['bias_q_a'] = None
            weights['w_q_a_norm'] = None
            weights['w_q_b'] = np.random.randn(hidden_size, num_q_heads * qk_head_dim).astype(self.dtype) * weight_scale
        
        # KV投影权重（关键修正：使用num_kv_heads）
        # KV的初始投影：输出包含低秩部分和旋转部分
        weights['w_kv_a'] = np.random.randn(hidden_size, kv_lora_rank + qk_rope_head_dim).astype(self.dtype) * weight_scale
        weights['bias_kv_a'] = np.zeros(kv_lora_rank + qk_rope_head_dim).astype(self.dtype) if self.has_bias else None
        weights['w_kv_a_norm'] = np.ones(kv_lora_rank).astype(self.dtype)  # RMSNorm权重
        
        # KV的低秩投影：输出维度为KV头数*(非旋转部分+值维度)
        weights['w_kv_b'] = np.random.randn(
            kv_lora_rank, 
            num_kv_heads * (qk_nope_head_dim + v_head_dim)  # 关键修正：使用num_kv_heads
        ).astype(self.dtype) * (1.0 / np.sqrt(kv_lora_rank))
        
        # 输出投影权重：输入维度为Q头数*值维度
        weights['w_o'] = np.random.randn(
            num_q_heads * v_head_dim, 
            hidden_size
        ).astype(self.dtype) * (1.0 / np.sqrt(num_q_heads * v_head_dim))
        weights['bias_o'] = np.zeros(hidden_size).astype(self.dtype) if self.has_bias else None
        
        return weights

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 写入输入输出张量
        test_writer.add_tensor(test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype))
        test_writer.add_tensor(test_writer.gguf_key("output"), self.output, raw_dtype=np_dtype_to_ggml(self.output.dtype))
        
        # 写入权重张量
        if self.weights['w_q_a'] is not None:
            test_writer.add_tensor(test_writer.gguf_key("w_q_a"), self.weights['w_q_a'], raw_dtype=np_dtype_to_ggml(self.weights['w_q_a'].dtype))
        if self.weights['bias_q_a'] is not None:
            test_writer.add_tensor(test_writer.gguf_key("bias_q_a"), self.weights['bias_q_a'], raw_dtype=np_dtype_to_ggml(self.weights['bias_q_a'].dtype))
        if self.weights['w_q_a_norm'] is not None:
            test_writer.add_tensor(test_writer.gguf_key("w_q_a_norm"), self.weights['w_q_a_norm'], raw_dtype=np_dtype_to_ggml(self.weights['w_q_a_norm'].dtype))
        if self.weights['w_q_b'] is not None:
            test_writer.add_tensor(test_writer.gguf_key("w_q_b"), self.weights['w_q_b'], raw_dtype=np_dtype_to_ggml(self.weights['w_q_b'].dtype))
            
        test_writer.add_tensor(test_writer.gguf_key("w_kv_a"), self.weights['w_kv_a'], raw_dtype=np_dtype_to_ggml(self.weights['w_kv_a'].dtype))
        if self.weights['bias_kv_a'] is not None:
            test_writer.add_tensor(test_writer.gguf_key("bias_kv_a"), self.weights['bias_kv_a'], raw_dtype=np_dtype_to_ggml(self.weights['bias_kv_a'].dtype))
        test_writer.add_tensor(test_writer.gguf_key("w_kv_a_norm"), self.weights['w_kv_a_norm'], raw_dtype=np_dtype_to_ggml(self.weights['w_kv_a_norm'].dtype))
        test_writer.add_tensor(test_writer.gguf_key("w_kv_b"), self.weights['w_kv_b'], raw_dtype=np_dtype_to_ggml(self.weights['w_kv_b'].dtype))
        test_writer.add_tensor(test_writer.gguf_key("w_o"), self.weights['w_o'], raw_dtype=np_dtype_to_ggml(self.weights['w_o'].dtype))
        if self.weights['bias_o'] is not None:
            test_writer.add_tensor(test_writer.gguf_key("bias_o"), self.weights['bias_o'], raw_dtype=np_dtype_to_ggml(self.weights['bias_o'].dtype))

        # 计算并写入标准答案
        ans = self._compute_reference_output()
        test_writer.add_tensor(test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64)

    def _compute_reference_output(self):
        """使用标准的DeepSeek V3 attention计算标准答案（修复RoPE生成的einsum错误）"""
        # 创建attention模块
        attention = create_attention_with_weights(self.config_dict, self.weights)
        
        # 解析输入形状
        batch_size, seq_len, hidden_size = self.input.shape
        
        # 转换输入为torch张量
        input_tensor = torch.from_numpy(self.input.astype(np.float64))
        
        # 生成位置ID
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_tensor.device).unsqueeze(0).expand(batch_size, -1)
        
        # 生成RoPE嵌入（匹配DeepseekV3RotaryEmbedding实现）
        rope_head_dim = self.config_dict["qk_rope_head_dim"]
        rope_theta = self.config_dict.get("rope_theta", 10000.0)
        
        # 计算频率（修复einsum维度不匹配问题）
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float64, device=input_tensor.device) / rope_head_dim))
        t = position_ids.unsqueeze(-1).float()  # 扩展为 (batch_size, seq_len, 1) 以匹配 einsum 需求
        freqs = torch.einsum("bse, e -> bse", t, inv_freq)  # 正确的维度映射: (batch, seq, 1) × (rope_dim//2) → (batch, seq, rope_dim//2)
        
        # 扩展为完整维度并计算cos/sin
        emb = torch.cat((freqs, freqs), dim=-1)  # (batch_size, seq_len, rope_head_dim)
        cos = emb.cos().to(dtype=torch.float64)
        sin = emb.sin().to(dtype=torch.float64)
        
        # 修正：RoPE不依赖batch维度，统一为(1, seq_len, rope_head_dim)以匹配原实现
        cos = cos[0:1, :, :]  # 取第一个batch的cos值用于广播
        sin = sin[0:1, :, :]
        
        # 创建因果注意力掩码 (batch_size, 1, seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_tensor.device))
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        attention_mask = torch.where(attention_mask, 0.0, float('-inf')).to(dtype=torch.float64)
        
        # 调用attention的forward方法
        with torch.no_grad():
            output, _ = attention.forward(
                hidden_states=input_tensor,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask,
                past_key_values=None,
                cache_position=None,  # 完整序列不需要cache_position
            )
        
        return output.numpy()


def gen_gguf(dtype: np.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []
    
    # 基础配置（符合DeepseekV3的典型参数）
    _BASE_CONFIG_ = {
        'num_attention_heads': 32,        # Q头数
        'num_key_value_heads': 8,         # KV头数（32/8=4个Q头共享一个KV头）
        'q_lora_rank': 512,               # Q的LoRA秩
        'kv_lora_rank': 512,              # KV的LoRA秩
        'qk_rope_head_dim': 64,           # QK中应用RoPE的维度
        'qk_nope_head_dim': 128,          # QK中不应用RoPE的维度
        'qk_head_dim': 192,               # 64+128（QK总维度）
        'v_head_dim': 128,                # V的维度
        'attention_dropout': 0.0,
        'rope_theta': 10000.0,
        'rope_interleave': False,         # 不使用交错RoPE
        '_attn_implementation': 'eager',  # 使用eager模式确保兼容性
        'max_position_embeddings': 4096,
        'rms_norm_eps': 1e-6,
        'rope_scaling': None,
    }

    # 测试用例配置：(input_shape, hidden_size, has_bias, use_q_lora, ...)
    _TEST_CASES_ = [
        # 基础测试（不同hidden_size和序列长度）
        ((2, 8, 512), 512, True, True, None, None, None, 48),
        ((2, 16, 768), 768, False, True, None, None, None, 49),
        ((2, 8, 512), 512, True, False, None, None, None, 50),
        
        # 长序列测试
        ((1, 32, 512), 512, True, True, None, None, None, 51),
        ((1, 64, 768), 768, False, False, None, None, None, 52),
        
        # 大batch测试
        ((4, 16, 512), 512, True, True, None, None, None, 53),
        
        # 自定义步长测试
        ((2, 8, 512), 512, True, True, [512*8, 512, 1], [512*8, 512, 1], None, 55),
    ]
    
    for input_shape, hidden_size, has_bias, use_q_lora, input_strides, output_strides, weight_strides, seed in _TEST_CASES_:
        config_dict = _BASE_CONFIG_.copy()
        config_dict['hidden_size'] = hidden_size
        
        test_case = MultiHeadLatentAttentionTestCase(
            input_shape=input_shape,
            config_dict=config_dict,
            input_strides=input_strides,
            output_strides=output_strides,
            weight_strides=weight_strides,
            has_bias=has_bias,
            use_q_lora=use_q_lora,
            dtype=dtype,
            seed=seed,
        )
        test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()

if __name__ == "__main__":
    _TENSOR_DTYPES_ = [
        np.float32, 
        np.float16, 
        bfloat16,
    ]

    dtype_filename_map = {
        np.float32: "multi_head_latent_attention_f32.gguf",
        np.float16: "multi_head_latent_attention_f16.gguf",
        bfloat16: "multi_head_latent_attention_bf16.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
    