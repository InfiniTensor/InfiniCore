from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def adaptive_avg_pool1d(input: Tensor, output_size: int) -> Tensor:
    r"""Apply a 1D adaptive average pooling."""
    
    # 直接调用底层绑定，传入 output_size
    # 注意：底层返回的是 impl 指针，需要包裹在 Tensor() 中
    return Tensor(_infinicore.adaptive_avg_pool1d(input._underlying, output_size))