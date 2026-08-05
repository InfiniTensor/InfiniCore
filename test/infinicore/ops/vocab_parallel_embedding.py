import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

import infinicore

VOCAB_START = 4
VOCAB_END = 12
HIDDEN_SIZE = 16
_ID_CASES = [
    torch.tensor([1, 4, 7, 11, 12, 19]),
    torch.tensor([[3, 4, 8], [11, 12, 15]]),
]
_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_INDEX_DTYPES = [infinicore.int32, infinicore.int64]


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("VocabParallelEmbedding")

    def get_test_cases(self):
        cases = []
        for ids in _ID_CASES:
            for index_dtype in _INDEX_DTYPES:
                for dtype in _DTYPES:
                    output_shape = (*ids.shape, HIDDEN_SIZE)
                    cases.append(
                        TestCase(
                            inputs=[
                                TensorSpec.from_tensor(
                                    tuple(ids.shape),
                                    dtype=index_dtype,
                                    init_mode=TensorInitializer.MANUAL,
                                    set_tensor=ids,
                                ),
                                TensorSpec.from_tensor(
                                    (VOCAB_END - VOCAB_START, HIDDEN_SIZE),
                                    dtype=dtype,
                                ),
                            ],
                            kwargs={
                                "vocab_start": VOCAB_START,
                                "vocab_end": VOCAB_END,
                            },
                            output_spec=TensorSpec.from_tensor(
                                output_shape, dtype=dtype
                            ),
                            comparison_target="out",
                            tolerance={"atol": 0, "rtol": 0},
                            description=(
                                f"indices {tuple(ids.shape)}, {index_dtype}, {dtype}"
                            ),
                        )
                    )
        return cases

    def torch_operator(self, indices, weight, vocab_start, vocab_end, *, out):
        mask = indices.lt(vocab_start).logical_or(indices.ge(vocab_end))
        local_indices = (indices - vocab_start).clamp(0, vocab_end - vocab_start - 1)
        result = F.embedding(local_indices, weight)
        result.masked_fill_(mask.unsqueeze(-1), 0)
        out.copy_(result)
        return out

    def infinicore_operator(self, indices, weight, vocab_start, vocab_end, *, out):
        return infinicore.vocab_parallel_embedding(
            indices, weight, vocab_start, vocab_end, out=out
        )


if __name__ == "__main__":
    GenericTestRunner(OpTest).run_and_exit()
