from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def var_mean(input, dim=None, unbiased=True, keepdim=False, out=None):
    """
    Calculates the variance and mean over the dimensions specified by dim. dim can be a single dimension, list of dimensions, or None to reduce over all dimensions.
    a = torch.tensor(
    [[ 0.2035,  1.2959,  1.8101, -0.4644],
     [ 1.5027, -0.3270,  0.5905,  0.6538],
     [-1.5745,  1.3330, -0.5596, -0.6548],
     [ 0.1264, -0.5080,  1.6420,  0.1992]]
)  # fmt: skip
torch.var_mean(a, dim=0, keepdim=True)
(tensor([[1.5926, 1.0056, 1.2005, 0.3646]]),
 tensor([[ 0.0645,  0.4485,  0.8707, -0.0665]]))
    """
    if out is None:
        var_tensor, mean_tensor = _infinicore.var_mean(input._underlying, dim, unbiased, keepdim)
        return (Tensor(var_tensor), Tensor(mean_tensor))
    var_output, mean_output = out
    _infinicore.var_mean_(var_output._underlying, mean_output._underlying, input._underlying, dim, unbiased, keepdim)

    return out
