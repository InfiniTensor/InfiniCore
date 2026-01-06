#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class RandomSampleBatched {
public:
    using schema = void (*)(Tensor, Tensor, const float *, const float *, const int *, const float *, int);
    static void execute(Tensor result, Tensor probs, const float *random_val, const float *topp, const int *topk, const float *temperature, int batch_size);
    static common::OpDispatcher<schema> &dispatcher();
};

// Out-of-place API
Tensor random_sample_batched(Tensor logits, const float *random_val, const float *topp, const int *topk, const float *temperature, int batch_size);
// In-place API
void random_sample_batched_(Tensor indices, Tensor logits, const float *random_val, const float *topp, const int *topk, const float *temperature, int batch_size);

} // namespace infinicore::op
