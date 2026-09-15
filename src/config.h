#ifndef INFINI_OPS_CONFIG_H_
#define INFINI_OPS_CONFIG_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "cloneable.h"

namespace infini::ops {

class Config {
 public:
  virtual ~Config() = default;

  virtual std::unique_ptr<Config> Clone() const {
    return std::make_unique<Config>(*this);
  }

  std::size_t implementation_index() const {
    return implementation_index_.value_or(0);
  }

  void set_implementation_index(std::size_t implementation_index) {
    implementation_index_ = implementation_index;
  }

  bool needs_implementation_resolution() const {
    return !implementation_index_.has_value();
  }

 private:
  std::optional<std::size_t> implementation_index_;
};

}  // namespace infini::ops

#endif
