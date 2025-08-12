from ast import List
import numpy as np
import gguf
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from transformers.models.deepseek_v3.modular_deepseek_v3 import DeepseekV3Attention
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def create_attention_with_weights(config_dict, weights):
    """创建attention模块并设置权重"""
    config = DeepseekV3Config(**config_dict)
    attention = DeepseekV3Attention(config, layer_idx=0)
    attention.eval()
    
    # 将随机生成的权重设置到模型中，先转换numpy为float64，再转torch
    if weights['w_q_a'] is not None:
        attention.q_a_proj.weight.data = torch.from_numpy(weights['w_q_a'].T.astype(np.float64))
        if weights['bias_q_a'] is not None:
            attention.q_a_proj.bias.data = torch.from_numpy(weights['bias_q_a'].astype(np.float64))
        attention.q_a_layernorm.weight.data = torch.from_numpy(weights['w_q_a_norm'].astype(np.float64))
        attention.q_b_proj.weight.data = torch.from_numpy(weights['w_q_b'].T.astype(np.float64))
    else:
        # 处理没有LoRA的情况
        attention.q_proj.weight.data = torch.from_numpy(weights['w_q_b'].T.astype(np.float64))
    
    attention.kv_a_proj_with_mqa.weight.data = torch.from_numpy(weights['w_kv_a'].T.astype(np.float64))
    if weights['bias_kv_a'] is not None:
        attention.kv_a_proj_with_mqa.bias.data = torch.from_numpy(weights['bias_kv_a'].astype(np.float64))
    attention.kv_a_layernorm.weight.data = torch.from_numpy(weights['w_kv_a_norm'].astype(np.float64))
    attention.kv_b_proj.weight.data = torch.from_numpy(weights['w_kv_b'].T.astype(np.float64))
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
        
        # 计算输出形状
        self.output_shape = input_shape[:-1] + (config_dict['hidden_size'],)
        
        # 验证配置
        self._validate_config()
        
        # 生成测试数据
        self._generate_test_data()

    def _validate_config(self):
        """验证配置的合理性"""
        assert self.config_dict['qk_head_dim'] == self.config_dict['qk_nope_head_dim'] + self.config_dict['qk_rope_head_dim'], \
            f"qk_head_dim({self.config_dict['qk_head_dim']}) should equal qk_nope_head_dim({self.config_dict['qk_nope_head_dim']}) + qk_rope_head_dim({self.config_dict['qk_rope_head_dim']})"
        assert self.config_dict['num_attention_heads'] % self.config_dict['num_key_value_heads'] == 0, \
            "num_attention_heads should be divisible by num_key_value_heads"
        assert self.config_dict['hidden_size'] == self.input_shape[-1], \
            f"hidden_size({self.config_dict['hidden_size']}) should match input last dimension({self.input_shape[-1]})"

    def _generate_test_data(self):
        """生成测试数据"""
        np.random.seed(self.seed)
        
        # 生成输入张量
        if self.input_strides is not None:
            self.input = np.random.randn(*self.input_shape).astype(self.dtype) * 0.1
            # TODO: 实现自定义步长的张量创建
        else:
            self.input = np.random.randn(*self.input_shape).astype(self.dtype) * 0.1
        
        # 生成权重
        self.weights = self._generate_weights()
        
        # 生成位置编码
        seq_len = self.input_shape[-2] if len(self.input_shape) >= 2 else 1
        self.cos_cache, self.sin_cache = self._generate_position_embeddings(seq_len)
        
        # 生成注意力掩码
        batch_size = self.input_shape[0] if len(self.input_shape) >= 2 else 1
        self.attention_mask = self._generate_attention_mask(batch_size, seq_len)
        
        # 初始化输出张量
        if self.output_strides is not None:
            self.output = np.empty(self.output_shape, dtype=self.dtype)
            # TODO: 实现自定义步长的张量创建
        else:
            self.output = np.empty(self.output_shape, dtype=self.dtype)

    def _generate_weights(self):
        """生成权重"""
        hidden_size = self.config_dict['hidden_size']
        num_heads = self.config_dict['num_attention_heads']
        q_lora_rank = self.config_dict.get('q_lora_rank')
        kv_lora_rank = self.config_dict['kv_lora_rank']
        qk_head_dim = self.config_dict['qk_head_dim']
        qk_nope_head_dim = self.config_dict['qk_nope_head_dim']
        qk_rope_head_dim = self.config_dict['qk_rope_head_dim']
        v_head_dim = self.config_dict['v_head_dim']
        
        # 使用更合理的初始化范围
        weight_scale = 1.0 / np.sqrt(hidden_size)
        
        weights = {}
        
        if q_lora_rank is not None:
            weights['w_q_a'] = np.random.randn(hidden_size, q_lora_rank).astype(self.dtype) * weight_scale
            weights['bias_q_a'] = np.zeros(q_lora_rank).astype(self.dtype) if self.has_bias else None
            weights['w_q_a_norm'] = np.ones(q_lora_rank).astype(self.dtype)
            weights['w_q_b'] = np.random.randn(q_lora_rank, num_heads * qk_head_dim).astype(self.dtype) * (1.0 / np.sqrt(q_lora_rank))
        else:
            weights['w_q_a'] = None
            weights['bias_q_a'] = None
            weights['w_q_a_norm'] = None
            weights['w_q_b'] = np.random.randn(hidden_size, num_heads * qk_head_dim).astype(self.dtype) * weight_scale
        
        weights['w_kv_a'] = np.random.randn(hidden_size, kv_lora_rank + qk_rope_head_dim).astype(self.dtype) * weight_scale
        weights['bias_kv_a'] = np.zeros(kv_lora_rank + qk_rope_head_dim).astype(self.dtype) if self.has_bias else None
        weights['w_kv_a_norm'] = np.ones(kv_lora_rank).astype(self.dtype)
        weights['w_kv_b'] = np.random.randn(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)).astype(self.dtype) * (1.0 / np.sqrt(kv_lora_rank))
        weights['w_o'] = np.random.randn(num_heads * v_head_dim, hidden_size).astype(self.dtype) * (1.0 / np.sqrt(num_heads * v_head_dim))
        weights['bias_o'] = np.zeros(hidden_size).astype(self.dtype) if self.has_bias else None
        
        return weights

    def _generate_position_embeddings(self, seq_len):
        """生成位置编码"""
        head_dim = self.config_dict['qk_rope_head_dim']
        rope_theta = self.config_dict['rope_theta']
        
        position_ids = np.arange(seq_len)
        freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
        angles = np.outer(position_ids, freqs)
        cos_vals = np.cos(angles).astype(self.dtype)
        sin_vals = np.sin(angles).astype(self.dtype)
        return cos_vals, sin_vals

    def _generate_attention_mask(self, batch_size, seq_len):
        """生成注意力掩码"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -np.inf
        return np.broadcast_to(mask[None, None, :, :], (batch_size, 1, seq_len, seq_len)).astype(self.dtype)

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
            
        # 写入位置编码和掩码
        test_writer.add_tensor(test_writer.gguf_key("cos_cache"), self.cos_cache, raw_dtype=np_dtype_to_ggml(self.cos_cache.dtype))
        test_writer.add_tensor(test_writer.gguf_key("sin_cache"), self.sin_cache, raw_dtype=np_dtype_to_ggml(self.sin_cache.dtype))
        test_writer.add_tensor(test_writer.gguf_key("attention_mask"), self.attention_mask, raw_dtype=np_dtype_to_ggml(self.attention_mask.dtype))

        # 写入配置参数
        for key, value in self.config_dict.items():
            if isinstance(value, (int, float, bool)):
                test_writer.add_config(test_writer.gguf_key(f"config_{key}"), value)
            elif isinstance(value, str):
                test_writer.add_config(test_writer.gguf_key(f"config_{key}"), value)

        # 计算并写入标准答案
        ans = self._compute_reference_output()
        test_writer.add_tensor(test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64)

    def _compute_reference_output(self):
        """调用已有库计算标准答案"""
        # 创建并配置attention模块
        attention = create_attention_with_weights(self.config_dict, self.weights)
        
        # 转换输入为torch张量
        input_tensor = torch.from_numpy(self.input.astype(np.float64))
        cos_tensor = torch.from_numpy(self.cos_cache.astype(np.float64))
        sin_tensor = torch.from_numpy(self.sin_cache.astype(np.float64))
        attention_mask_tensor = torch.from_numpy(self.attention_mask.astype(np.float64))
        
        # 调用forward方法
        with torch.no_grad():
            output, _ = attention.forward(
                hidden_states=input_tensor,
                position_embeddings=(cos_tensor, sin_tensor),
                attention_mask=attention_mask_tensor,
                past_key_value=None,
                cache_position=None,
            )
        
        return output.numpy()

def gen_gguf(dtype: np.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []
    # 基础配置
    _BASE_CONFIG_ = {
        'num_attention_heads': 32,
        'num_key_value_heads': 8,
        'q_lora_rank': 512,
        'kv_lora_rank': 512,
        'qk_rope_head_dim': 64,
        'qk_nope_head_dim': 128,
        'qk_head_dim': 192,  # qk_nope_head_dim + qk_rope_head_dim
        'v_head_dim': 128,
        'attention_dropout': 0.0,
        'rope_theta': 10000.0,
        'rope_interleave': False,
        '_attn_implementation': 'eager',
        'max_position_embeddings': 4096,
        'rms_norm_eps': 1e-6,
        'rope_scaling': None,
    }

    # 测试用例配置：(input_shape, hidden_size, has_bias, use_q_lora, input_strides, output_strides, weight_strides, seed)
    _TEST_CASES_ = [
        # 基本测试：2D输入，带偏置，使用Q LoRA
        ((3, 512), 512, True, True, None, None, None, 42),
        ((5, 768), 768, True, True, None, None, None, 43),
        
        # 基本测试：2D输入，不带偏置，使用Q LoRA
        ((3, 512), 512, False, True, None, None, None, 44),
        ((5, 768), 768, False, True, None, None, None, 45),
        
        # 基本测试：2D输入，带偏置，不使用Q LoRA
        ((3, 512), 512, True, False, None, None, None, 46),
        ((5, 768), 768, True, False, None, None, None, 47),
        
        # 3D输入测试：batch_size=2
        ((2, 8, 512), 512, True, True, None, None, None, 48),
        ((2, 16, 768), 768, False, True, None, None, None, 49),
        ((2, 8, 512), 512, True, False, None, None, None, 50),
        
        # 更长序列测试
        ((1, 32, 512), 512, True, True, None, None, None, 51),
        ((1, 64, 768), 768, False, False, None, None, None, 52),
        
        # 大batch测试
        ((4, 16, 512), 512, True, True, None, None, None, 53),
        ((8, 8, 768), 768, False, True, None, None, None, 54),
        
        # 自定义步长测试 (TODO: 需要实现自定义步长支持)
        # ((2, 8, 512), 512, True, True, [8*512*4, 512*4, 4], None, None, 55),
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

    print(f"Generated {len(test_cases)} test cases")
    test_writer.add_tests(test_cases)
    test_writer.save()
    print("Test cases saved to multi_head_latent_attention.gguf")

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