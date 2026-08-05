from infinicore.lib import _infinicore


def vocab_parallel_embedding(indices, weight, vocab_start, vocab_end, *, out):
    _infinicore.vocab_parallel_embedding_(
        out._underlying,
        indices._underlying,
        weight._underlying,
        vocab_start,
        vocab_end,
    )
    return out
