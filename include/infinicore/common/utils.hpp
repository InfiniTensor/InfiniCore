#ifndef UTILS_HPP
#define UTILS_HPP

#include <functional>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <type_traits>

#include "../tensor.hpp"
#include "infinicore.h"

inline struct SpdlogInitializer {
    SpdlogInitializer() {
        if (!std::getenv("INFINICORE_LOG_LEVEL")) {
            spdlog::set_level(spdlog::level::off);
        } else {
            spdlog::cfg::load_env_levels("INFINICORE_LOG_LEVEL");
        }
    }
} spdlog_initializer;

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

#define INFINICORE_CHECK_ERROR(call)                                                           \
    do {                                                                                       \
        spdlog::info("Entering `" #call "` at `" __FILE__ ":" STRINGIZE(__LINE__) "`.");       \
        int ret = (call);                                                                      \
        spdlog::info("Exiting `" #call "` at `" __FILE__ ":" STRINGIZE(__LINE__) "`.");        \
        if (ret != INFINI_STATUS_SUCCESS) {                                                    \
            throw std::runtime_error(#call " failed with error code: " + std::to_string(ret)); \
        }                                                                                      \
    } while (false)

namespace infinicore {

// Base hash_combine for arithmetic types
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, void>
hash_combine(size_t &seed, const T &value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Specialization for Tensor
inline void hash_combine(size_t &seed, Tensor tensor) {
    hash_combine(seed, static_cast<size_t>(tensor->dtype()));
    for (Size shape : tensor->shape()) {
        hash_combine(seed, shape);
    }
    for (Stride stride : tensor->strides()) {
        hash_combine(seed, static_cast<size_t>(stride));
    }
}

// Specialization for std::string
inline void hash_combine(size_t &seed, const std::string &str) {
    hash_combine(seed, std::hash<std::string>{}(str));
}

// Specialization for const char*
inline void hash_combine(size_t &seed, const char *str) {
    hash_combine(seed, std::string(str));
}

// Variadic template for multiple arguments
template <typename First, typename... Rest>
void hash_combine(size_t &seed, const First &first, const Rest &...rest) {
    hash_combine(seed, first);
    hash_combine(seed, rest...);
}

// Base case for variadic template
inline void hash_combine(size_t &seed) {
    // Base case - do nothing
}

// Convenience function to hash multiple values
template <typename... Types>
size_t hash_combine(const Types &...values) {
    size_t seed = 0;
    hash_combine(seed, values...);
    return seed;
}

} // namespace infinicore

#endif // UTILS_HPP
