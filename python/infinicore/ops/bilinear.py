from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def bilinear(input1, input2, weight, bias=None, *, out=None):
    if out is None:
        if bias is None:
            return Tensor(_infinicore.bilinear(input1._underlying,
                                              input2._underlying,
                                              weight._underlying))
        else:
            return Tensor(_infinicore.bilinear_bias(input1._underlying,
                                                   input2._underlying,
                                                   weight._underlying,
                                                   bias._underlying))

    if bias is None:
        _infinicore.bilinear_(out._underlying,
                              input1._underlying,
                              input2._underlying,
                              weight._underlying)
    else:
        _infinicore.bilinear_bias_(out._underlying,
                                   input1._underlying,
                                   input2._underlying,
                                   weight._underlying,
                                   bias._underlying)

    return out