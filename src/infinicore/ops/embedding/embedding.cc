#include "infinicore/ops/embedding.hpp"
#include "infinicore/context/context.hpp"
#include <cstring>

namespace infinicore::op {

Tensor embedding(Tensor input, // LongTensor of arbitrary shape containing the indices to extract
                 Tensor weight // Weight: Embedding matrix of floating point type with shape (V, embedding_dim), where V = maximum index + 1
) {
    auto input_shape = input->shape();
    auto weight_shape = weight->shape();
    auto vocab_size = weight_shape[0];
    auto embedding_dim = weight_shape[1];

    // Assign memory to out variables
    auto output_shape = input_shape;
    output_shape.push_back(embedding_dim);
    Tensor inputs_embeds = Tensor::empty(output_shape, weight->dtype(), weight->device());

    embedding_(inputs_embeds, input, weight);
    return inputs_embeds;
}

void embedding_(Tensor out, Tensor input, Tensor weight) {
    assert(infinicore::DataType::I64 == input->dtype() || (infinicore::DataType::I32 == input->dtype()));
    assert(infinicore::Device::Type::CPU == input->device());

    auto input_shape = input->shape();
    auto weight_shape = weight->shape();
    auto vocab_size = weight_shape[0];
    auto embedding_dim = weight_shape[1];

    // Calculate the number of token
    Size counts = 1;
    for (auto &v : input_shape) {
        counts *= v;
    }

    // the bytes of one token
    const Size bytes = dsize(weight->dtype()) * embedding_dim;
    auto *weight_ptr = weight->data();
    auto *out_ptr = out->data();

    // copies
    if (weight->device().getType() == Device::Type::CPU) {
        if (infinicore::DataType::I64 == input->dtype()) {
            const int64_t *input_arr = reinterpret_cast<const int64_t *>(input->data());
            for (Size i = 0; i < counts; ++i) {
                int64_t idx = input_arr[i];
                assert((idx >= 0) && (idx < vocab_size));
                std::memcpy(out_ptr + i * bytes,
                            weight_ptr + idx * bytes,
                            bytes);
            }
        } else if (infinicore::DataType::I32 == input->dtype()) {
            const int32_t *input_arr = reinterpret_cast<const int32_t *>(input->data());

            for (Size i = 0; i < counts; ++i) {
                int32_t idx = input_arr[i];
                assert((idx >= 0) && (idx < vocab_size));
                std::memcpy(out_ptr + i * bytes,
                            weight_ptr + idx * bytes,
                            bytes);
            }
        }

    } else {
        if (infinicore::DataType::I64 == input->dtype()) {
            const int64_t *input_arr = reinterpret_cast<const int64_t *>(input->data());
            for (Size i = 0; i < counts; ++i) {
                int64_t idx = input_arr[i];
                assert((idx >= 0) && (idx < vocab_size));
                context::memcpyD2D(out_ptr + i * bytes,
                                   weight_ptr + idx * bytes,
                                   bytes);
            }
        } else if (infinicore::DataType::I32 == input->dtype()) {
            const int32_t *input_arr = reinterpret_cast<const int32_t *>(input->data());
            for (Size i = 0; i < counts; ++i) {
                int32_t idx = input_arr[i];
                assert((idx >= 0) && (idx < vocab_size));
                context::memcpyD2D(out_ptr + i * bytes,
                                   weight_ptr + idx * bytes,
                                   bytes);
            }
        }
    }
}

} // namespace infinicore::op
