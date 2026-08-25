import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase

import infinicore


_TIMESTEP_SHAPES = [(1,), (4,), (17,)]
_EMBEDDING_DIMS = [16, 256]
_INPUT_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("TimestepEmbedding")

    def get_test_cases(self):
        cases = []
        for shape in _TIMESTEP_SHAPES:
            for embedding_dim in _EMBEDDING_DIMS:
                for dtype in _INPUT_DTYPES:
                    cases.append(
                        TestCase(
                            inputs=[TensorSpec.from_tensor(shape, None, dtype)],
                            kwargs={
                                "embedding_dim": embedding_dim,
                                "max_period": 10000.0,
                            },
                            output_spec=None,
                            comparison_target=None,
                            tolerance={"atol": 2e-5, "rtol": 2e-5},
                            description="TimestepEmbedding - OUT_OF_PLACE",
                        )
                    )
        return cases

    def torch_operator(
        self,
        timestep,
        embedding_dim=256,
        max_period=10000.0,
        out=None,
    ):
        half_dim = embedding_dim // 2
        exponent = -torch.log(torch.tensor(max_period)) * torch.arange(
            half_dim,
            dtype=torch.float32,
            device=timestep.device,
        ) / half_dim
        angles = timestep.float().unsqueeze(1) * exponent.exp().unsqueeze(0)
        result = torch.cat((angles.cos(), angles.sin()), dim=1)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(
        self,
        timestep,
        embedding_dim=256,
        max_period=10000.0,
        out=None,
    ):
        return infinicore.timestep_embedding(
            timestep,
            embedding_dim,
            max_period,
            out=out,
        )


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
