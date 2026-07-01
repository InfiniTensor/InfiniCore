#pragma once

#include "../context/context.hpp"
#include "../tensor.hpp"
#include "module.hpp"
#include <array>
#include <string>
#include <utility>

namespace infinicore::nn {

class MRoPE : public Module {
public:
    MRoPE(size_t head_dim,
          size_t rotary_dim,
          size_t max_seq_len,
          double theta,
          std::array<int, 3> section,
          bool interleaved,
          const DataType &dtype,
          const Device &device);

    std::pair<Tensor, Tensor> forward(const Tensor &q,
                                      const Tensor &k,
                                      const Tensor &positions) const;

    std::pair<Tensor, Tensor> forward(const Tensor &q_out,
                                      const Tensor &k_out,
                                      const Tensor &q,
                                      const Tensor &k,
                                      const Tensor &positions) const;

    size_t rotary_dim() const { return rotary_dim_; }
    size_t head_dim() const { return head_dim_; }
    size_t max_seq_len() const { return max_seq_len_; }
    double theta() const { return theta_; }
    const std::array<int, 3> &section() const { return section_; }
    bool interleaved() const { return interleaved_; }
    DataType dtype() const { return dtype_; }

    std::string extra_repr() const;

protected:
    INFINICORE_NN_BUFFER(sin_cache);
    INFINICORE_NN_BUFFER(cos_cache);

private:
    void initialize_cache();

    size_t head_dim_;
    size_t rotary_dim_;
    size_t max_seq_len_;
    double theta_;
    std::array<int, 3> section_;
    bool interleaved_;
    DataType dtype_;
};

} // namespace infinicore::nn
