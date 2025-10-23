#pragma once

#include "../tensor.hpp"

namespace infinicore::nn {
class Parameter : public Tensor {
public:
    Parameter(const Shape &shape,
              const DataType &dtype,
              const Device &device);

    void load_blob(const void *data);
};
} // namespace infinicore::nn
