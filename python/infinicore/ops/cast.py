from infinicore.lib import _infinicore


def cast(input, *, out):
    _infinicore.cast_(out._underlying, input._underlying)
    return out
