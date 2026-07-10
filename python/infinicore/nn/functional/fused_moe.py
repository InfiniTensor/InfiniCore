from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

ACT_SILU = 0
ACT_SWIGLU = 1


def fused_moe(
    input: Tensor,
    token_selected_experts: Tensor,
    token_final_scales: Tensor,
    w1: Tensor,
    w2: Tensor,
    *,
    b1: Tensor | None = None,
    b2: Tensor | None = None,
    activation: int = ACT_SWIGLU,
    out: Tensor | None = None,
) -> Tensor:
    b1_arg = None if b1 is None else b1._underlying
    b2_arg = None if b2 is None else b2._underlying
    if out is None:
        return Tensor(
            _infinicore.fused_moe(
                input._underlying,
                token_selected_experts._underlying,
                token_final_scales._underlying,
                w1._underlying,
                w2._underlying,
                b1_arg,
                b2_arg,
                activation,
            )
        )

    _infinicore.fused_moe_(
        out._underlying,
        input._underlying,
        token_selected_experts._underlying,
        token_final_scales._underlying,
        w1._underlying,
        w2._underlying,
        b1_arg,
        b2_arg,
        activation,
    )
    return out
