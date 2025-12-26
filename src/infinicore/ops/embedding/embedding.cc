#include "infinicore/ops/embedding.hpp"
#include "infinicore/context/context.hpp"
#include "../../utils.hpp"
#include <cstring>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Embedding::schema> &Embedding::dispatcher() {
    static common::OpDispatcher<Embedding::schema> dispatcher_;
    return dispatcher_;
}

void Embedding::execute(Tensor out, Tensor input, Tensor weight) {
    // Check that output and weight are on the same device
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, weight);
    
    // Set device context
    infinicore::context::setDevice(out->device());
    
    // Use dispatcher to lookup kernel (infiniop implementation)
    dispatcher().lookup(out->device().getType())(out, input, weight);
}

Tensor embedding(Tensor input, // LongTensor of arbitrary shape containing the indices to extract
                 Tensor weight // Weight: Embedding matrix of floating point type with shape (V, embedding_dim), where V = maximum index + 1
) {
    auto input_shape = input->shape();
    auto weight_shape = weight->shape();
    auto embedding_dim = weight_shape[1];

    // Assign memory to out variables
    auto output_shape = input_shape;
    output_shape.push_back(embedding_dim);
    Tensor inputs_embeds = Tensor::empty(output_shape, weight->dtype(), weight->device());

    embedding_(inputs_embeds, input, weight);
    return inputs_embeds;
}

void embedding_(Tensor out, Tensor input, Tensor weight) {
    Embedding::execute(out, input, weight);
}

} // namespace infinicore::op
