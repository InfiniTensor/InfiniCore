import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

import infinicore

_CASES = [
    (infinicore.float32, infinicore.float16),
    (infinicore.float16, infinicore.float32),
    (infinicore.bfloat16, infinicore.float32),
    (infinicore.int32, infinicore.float32),
]


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Cast")

    def get_test_cases(self):
        cases = []
        for input_dtype, output_dtype in _CASES:
            init_mode = (
                TensorInitializer.RANDINT
                if input_dtype == infinicore.int32
                else TensorInitializer.RANDOM
            )
            cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor(
                            (4, 7),
                            dtype=input_dtype,
                            init_mode=init_mode,
                            low=-32,
                            high=32,
                        )
                    ],
                    output_spec=TensorSpec.from_tensor((4, 7), dtype=output_dtype),
                    comparison_target="out",
                    tolerance={"atol": 0, "rtol": 0},
                    description=f"cast {input_dtype} to {output_dtype}",
                )
            )
        return cases

    def torch_operator(self, input, *, out):
        out.copy_(input)
        return out

    def infinicore_operator(self, input, *, out):
        return infinicore.cast(input, out=out)


if __name__ == "__main__":
    GenericTestRunner(OpTest).run_and_exit()
