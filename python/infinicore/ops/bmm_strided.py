from infinicore.lib import _infinicore


def bmm_strided(input, mat2, *, out):
    _infinicore.bmm_strided_(out._underlying, input._underlying, mat2._underlying)
    return out
