import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

import infinicore


def torch_w4a8_moe_shuffle(packed_weight):
    n, packed_k = packed_weight.shape
    values = packed_weight.to(torch.uint8)
    unpacked = torch.stack((values >> 4, values & 0x0F), dim=-1).reshape(n, -1)
    blocks = unpacked.reshape(n, -1, 8)
    blocked = ((blocks[..., :4] << 4) | blocks[..., 4:]).reshape(n, packed_k)

    n_tile = 256 if n % 256 == 0 else n
    shuffled = blocked.t().reshape(packed_k // 32, 32, n // n_tile, n_tile)
    shuffled = shuffled.permute(0, 2, 3, 1).contiguous()
    shuffled = shuffled.reshape(
        packed_k // 32, n // n_tile, n_tile // 32, 32, 4, 8
    )
    return (
        shuffled.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(n, packed_k)
        .to(torch.int8)
    )


def make_cases():
    generator = torch.Generator(device="cpu").manual_seed(20260911)
    cases = []
    for n, logical_k in ((96, 128), (256, 256), (512, 128)):
        packed = torch.randint(
            -128,
            128,
            (n, logical_k // 2),
            generator=generator,
            dtype=torch.int8,
        )
        spec = TensorSpec.from_tensor(
            tuple(packed.shape),
            None,
            infinicore.int8,
            init_mode=TensorInitializer.MANUAL,
            set_tensor=packed,
            name="packed_weight",
        )
        cases.append(
            TestCase(
                inputs=[spec],
                kwargs={},
                output_spec=None,
                comparison_target=None,
                tolerance={"atol": 0, "rtol": 0},
                description=f"w4a8_moe_shuffle - N={n}, K={logical_k}",
            )
        )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("w4a8_moe_shuffle")

    def get_test_cases(self):
        return make_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_w4a8_moe_shuffle(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.nn.functional.w4a8_moe_shuffle(*args, **kwargs)


if __name__ == "__main__":
    GenericTestRunner(OpTest).run_and_exit()
