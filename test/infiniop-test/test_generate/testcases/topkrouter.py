import numpy as np
from typing import List
import torch
import torch.nn as nn

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides


def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, shape).astype(dtype) * 0.001


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, correction_bias,
                 routed_scaling_factor: float,
                 topk: int, config=None):
        super().__init__()
        self.config = config
        self.top_k = 8  # config.num_experts_per_tok
        self.n_routed_experts = 256  # config.n_routed_experts
        self.routed_scaling_factor = 2.5  # config.routed_scaling_factor
        self.n_group = 8  # config.n_group
        self.topk_group = 4  # config.topk_group
        self.norm_topk_prob = True  # config.norm_topk_prob

        self.routed_scaling_factor = routed_scaling_factor
        self.top_k = topk

        # self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        # self.weight = torch.rand(256, 7168) * 2 - 1

        # self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        self.e_score_correction_bias = torch.zeros(256, )
        self.e_score_correction_bias[:] = correction_bias[:]

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)  # Size([1, 256])
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )

        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=True)[1]  # Size([1, 4])
        group_mask = torch.zeros_like(group_scores)  # Size([1, 8])
        group_mask.scatter_(1, group_idx, 1)  # Size([1, 8])

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )

        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # Size([1, 256])
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=True)[1]  # Size([1, 8])

        return topk_indices

    def forward(self, router_logits):
        # hidden_states = hidden_states.view(-1, 7168)
        # router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = router_logits.sigmoid()  # (1,256)
        scores = scores.to(torch.float32)

        topk_indices = self.get_topk_indices(scores)  # (1,8)
        topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


def python_topkrouter(x: np.ndarray,
                      correction_bias: np.ndarray,
                      routed_scaling_factor: float,
                      topk: int):
    x = torch.from_numpy(x)
    correction_bias = torch.from_numpy(correction_bias)

    router_logits = x
    lable_indices, lable_values = DeepseekV3TopkRouter(correction_bias, routed_scaling_factor=routed_scaling_factor, topk=topk)(router_logits)
    lable_indices = lable_indices.to(torch.int32)

    return lable_values.numpy(), lable_indices.numpy()


class TopkrouterTestCase(InfiniopTestCase):
    def __init__(self,
                 values: np.ndarray,  # 传出参数
                 indices: np.ndarray,  # 传出参数
                 x: np.ndarray,  # 传入参数
                 correction_bias: np.ndarray,  # 传入参数
                 routed_scaling_factor: float,
                 topk: int,
                 values_shape: List[int] | None,
                 values_strides: List[int] | None,
                 indices_shape: List[int] | None,
                 indices_strides: List[int] | None,
                 x_shape: List[int] | None,
                 x_strides: List[int] | None,
                 correction_bias_shape: List[int] | None,
                 correction_bias_strides: List[int] | None,
                 ):
        super().__init__("topkrouter")
        self.values = values
        self.indices = indices
        self.x = x
        self.correction_bias = correction_bias

        self.routed_scaling_factor = routed_scaling_factor
        self.topk = topk

        self.values_shape = values_shape
        self.values_strides = values_strides
        self.indices_shape = indices_shape
        self.indices_strides = indices_strides
        self.x_shape = x_shape
        self.x_strides = x_strides
        self.correction_bias_shape = correction_bias_shape
        self.correction_bias_strides = correction_bias_strides

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        if self.values_shape is not None:
            print("self.values_shape:  ", self.values_shape)
            test_writer.add_array(test_writer.gguf_key("values.shape"), self.values_shape)
        if self.indices_shape is not None:
            test_writer.add_array(test_writer.gguf_key("indices.shape"), self.indices_shape)
        if self.x_shape is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.x_shape)
        if self.correction_bias_shape is not None:
            test_writer.add_array(test_writer.gguf_key("correction_bias.shape"), self.correction_bias_shape)

        if self.x_strides is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.x_strides))
        if self.correction_bias_strides is not None:
            test_writer.add_array(test_writer.gguf_key("correction_bias_strides.strides"), gguf_strides(*self.correction_bias_strides))
        test_writer.add_array(
            test_writer.gguf_key("values.strides"),
            gguf_strides(*self.values_strides if self.values_strides is not None else contiguous_gguf_strides(self.values_shape))
        )
        test_writer.add_array(
            test_writer.gguf_key("indices.strides"),
            gguf_strides(*self.indices_strides if self.indices_strides is not None else contiguous_gguf_strides(self.indices_shape))
        )

        test_writer.add_tensor(test_writer.gguf_key("values"),
                               self.values,
                               raw_dtype=np_dtype_to_ggml(self.values.dtype))

        test_writer.add_tensor(test_writer.gguf_key("indices"),
                               self.indices,
                               raw_dtype=np_dtype_to_ggml(self.indices.dtype))

        test_writer.add_tensor(test_writer.gguf_key("x"),
                               self.x,
                               raw_dtype=np_dtype_to_ggml(self.x.dtype))

        test_writer.add_tensor(test_writer.gguf_key("correction_bias"),
                               self.correction_bias,
                               raw_dtype=np_dtype_to_ggml(self.correction_bias.dtype))

        test_writer.add_float32(test_writer.gguf_key("routed_scaling_factor"), self.routed_scaling_factor)
        test_writer.add_int32(test_writer.gguf_key("topk"), self.topk)

        lable_values, lable_indices = python_topkrouter(self.x.copy(), self.correction_bias.copy(), self.routed_scaling_factor, self.topk)

        test_writer.add_tensor(
            test_writer.gguf_key("lable_values"),
            lable_values,
            raw_dtype=np_dtype_to_ggml(lable_values.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("lable_indices"),
            lable_indices,
            raw_dtype=np_dtype_to_ggml(lable_indices.dtype)
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("topkrouter.gguf")
    test_cases = []

    _TEST_CASES_ = [
        # x_shape, x_strides, correction_bias_shape, correction_bias_stride， routed_scaling_factor, topk
        ((1, 256), None, (256,), None, 2.5, 8),
        ((2, 256), None, (256,), None, 1.5, 8),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for x_shape, x_strides, correction_bias_shape, b_stride, routed_scaling_factor, topk in _TEST_CASES_:
            ntoken = x_shape[0]

            values_indices_shape = (ntoken, topk)
            values = np.empty(tuple(0 for _ in values_indices_shape), dtype=np.float32)
            indices = np.empty(tuple(0 for _ in values_indices_shape), dtype=np.int32)

            x = np.random.rand(*x_shape).astype(dtype)
            correction_bias = np.random.rand(*correction_bias_shape).astype(np.float32)
            test_case = TopkrouterTestCase(
                values=values,
                indices=indices,
                x=x,
                correction_bias=correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                topk=topk,
                values_shape=list(values_indices_shape),
                values_strides=None,
                indices_shape=list(values_indices_shape),
                indices_strides=None,
                x_shape=list(x_shape),
                x_strides=None,
                correction_bias_shape=list(correction_bias_shape),
                correction_bias_strides=None,
            )

            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
