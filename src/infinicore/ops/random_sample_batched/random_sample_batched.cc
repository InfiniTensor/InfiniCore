#include "infinicore/ops/random_sample_batched.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<RandomSampleBatched::schema> &RandomSampleBatched::dispatcher() {
    static common::OpDispatcher<RandomSampleBatched::schema> dispatcher_;
    return dispatcher_;
};

void RandomSampleBatched::execute(
    Tensor result,
    Tensor probs,
    const float *random_val,
    const float *topp,
    const int *topk,
    const float *temperature,
    int batch_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(result, probs);
    infinicore::context::setDevice(result->device());
    auto device_type = result->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No RandomSampleBatched implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(result, probs, random_val, topp, topk, temperature, batch_size);
}

Tensor random_sample_batched(
    Tensor logits,
    const float *random_val,
    const float *topp,
    const int *topk,
    const float *temperature,
    int batch_size) {
    Shape shape = logits->shape();
    auto result = Tensor::empty(shape, DataType::I32, logits->device());
    random_sample_batched_(result, logits, random_val, topp, topk, temperature, batch_size);
    return result;
}
void random_sample_batched_(
    Tensor result,
    Tensor logits,
    const float *random_val,
    const float *topp,
    const int *topk,
    const float *temperature,
    int batch_size) {
    RandomSampleBatched::execute(result, logits, random_val, topp, topk, temperature, batch_size);
}
} // namespace infinicore::op
