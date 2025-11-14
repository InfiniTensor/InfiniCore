import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.tensor import TensorInitializer
from framework.runner import GenericTestRunner

# Test cases format: (num_embeddings, embedding_dim, indices_shape, padding_idx_or_None)
# Embedding typically uses contiguous weight tensors; we include different sizes and padding_idx.
_TEST_CASES_DATA = [
    (10, 4, (3, 5), None),
    (20, 8, (6,), None),
    (5, 3, (2, 2), 0),
    (15, 6, (4, 3), None),
    (7, 7, (1, 10), None),
    (12, 5, (3, 3), 1),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for num_embeddings, emb_dim, idx_shape, padding_idx in _TEST_CASES_DATA:
        # weight is (num_embeddings, emb_dim)
        weight_spec = TensorSpec.from_tensor(
            (num_embeddings, emb_dim), None, infinicore.float32
        )
        indices_spec = TensorSpec.from_tensor(
            idx_shape,
            None,
            infinicore.int64,
            init_mode=TensorInitializer.RANDINT,
            low=0,
            high=num_embeddings,  # infinicore.randint high is exclusive, so this yields 0..num_embeddings-1
        )

        kwargs = {}
        if padding_idx is not None:
            kwargs["padding_idx"] = padding_idx

        test_cases.append(
            TestCase(
                inputs=[weight_spec, indices_spec],
                kwargs=kwargs,
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"embedding - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """Embedding operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Embedding")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.embedding(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.embedding(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
