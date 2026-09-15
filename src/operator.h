#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config.h"
#include "dispatcher.h"
#include "handle.h"
#include "host_range_profiler.h"
#include "runtime.h"
#include "tensor.h"
#include "tuning.h"

namespace infini::ops::detail {

struct CacheKey {
  std::size_t hash;

  std::vector<Tensor> tensors;

  std::size_t scalar_hash;

  template <typename... Args>
  static CacheKey Build(const Args&... args) {
    CacheKey key;
    key.hash = 0;
    key.scalar_hash = 0;
    (key.Absorb(args), ...);
    return key;
  }

 private:
  void Absorb(const Tensor& t) {
    HashCombine(hash, t);
    tensors.push_back(t);
  }

  void Absorb(const std::vector<Tensor>& ts) {
    HashCombine(hash, ts.size());
    for (const auto& t : ts) {
      HashCombine(hash, t);
      tensors.push_back(t);
    }
  }

  template <typename T>
  void Absorb(const T& v) {
    HashCombine(hash, v);
    HashCombine(scalar_hash, v);
  }
};

template <typename Functor, typename... Args, auto... implementation_indices>
auto DispatchImplementation(std::size_t implementation_index, Functor&& func,
                            std::string_view context_str,
                            List<implementation_indices...>, Args&&... args) {
  return DispatchFunc<std::size_t,
                      static_cast<std::size_t>(implementation_indices)...>(
      implementation_index, std::forward<Functor>(func), context_str,
      std::forward<Args>(args)...);
}

template <auto... values>
std::vector<std::size_t> ListToVector(List<values...>) {
  return {static_cast<std::size_t>(values)...};
}

template <typename ValueType, auto... values>
bool ListContains(ValueType value, List<values...>) {
  return ((value == static_cast<ValueType>(values)) || ...);
}

inline void SyncDevice(Device::Type dev_type) {
  DispatchFunc<ActiveDevices<void>>(
      dev_type,
      [](auto device_tag) {
        constexpr Device::Type kDev = decltype(device_tag)::value;
        infini::rt::runtime::Runtime<kDev>::DeviceSynchronize();
      },
      "SyncDevice");
}

inline Device::Type FirstDeviceType() { return Device::Type::kCount; }

template <typename First, typename... Rest>
Device::Type FirstDeviceType(const First& first, const Rest&... rest) {
  if constexpr (std::is_same_v<std::decay_t<First>, Tensor>) {
    return first.device().type();
  } else if constexpr (std::is_same_v<std::decay_t<First>,
                                      std::vector<Tensor>>) {
    return first.empty() ? FirstDeviceType(rest...)
                         : first.front().device().type();
  } else {
    return FirstDeviceType(rest...);
  }
}

template <typename TensorLike, typename = void>
class IsTensorLike : public std::false_type {};

template <typename TensorLike>
class IsTensorLike<
    TensorLike,
    std::void_t<decltype(std::declval<const TensorLike&>().data()),
                decltype(std::declval<const TensorLike&>().shape()),
                decltype(std::declval<const TensorLike&>().strides()),
                decltype(std::declval<const TensorLike&>().dtype()),
                decltype(std::declval<const TensorLike&>().device())>>
    : public std::true_type {};

template <typename T, typename std::enable_if_t<
                          IsTensorLike<std::decay_t<T>>::value, int> = 0>
Tensor AsCallArg(const T& tensor) {
  return Tensor{tensor};
}

template <typename T, typename std::enable_if_t<
                          !IsTensorLike<std::decay_t<T>>::value, int> = 0>
const T& AsCallArg(const T& value) {
  return value;
}

template <typename Key, typename TensorLike, typename Args, typename = void>
class HasMakeReturnValueImpl : public std::false_type {};

template <typename Key, typename TensorLike, typename... Args>
class HasMakeReturnValueImpl<
    Key, TensorLike, std::tuple<Args...>,
    std::void_t<decltype(Key::MakeReturnValue(std::declval<const TensorLike&>(),
                                              std::declval<const Args&>()...))>>
    : public std::true_type {};

template <typename Key, typename... Args>
class HasMakeReturnValueImpl<Key, Tensor, std::tuple<Args...>>
    : public std::false_type {};

template <typename Key, typename TensorLike, typename... Args>
class HasMakeReturnValue
    : public HasMakeReturnValueImpl<Key, std::decay_t<TensorLike>,
                                    std::tuple<Args...>> {};

inline bool TraceOperatorCallsEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("INFINI_OPS_TRACE_CALLS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

template <typename Key>
constexpr std::string_view OperatorName() {
#if defined(__clang__) || defined(__GNUC__)
  std::string_view name{__PRETTY_FUNCTION__};
  constexpr std::string_view marker{"Key = "};
  const auto start = name.find(marker) + marker.size();
  const auto end = name.find_first_of(";]", start);
  name = name.substr(start, end - start);
#else
  std::string_view name{"unknown"};
#endif
  constexpr std::string_view namespace_prefix{"infini::ops::"};
  if (name.rfind(namespace_prefix, 0) == 0) {
    name.remove_prefix(namespace_prefix.size());
  }
  return name;
}

template <typename Key>
void TraceOperatorCall(const CacheKey& key, const Config& config) {
  if (!TraceOperatorCallsEnabled()) return;

  const auto device =
      key.tensors.empty()
          ? std::string_view{"unknown"}
          : Device::StringFromType(key.tensors.front().device().type());
  constexpr auto operator_name = OperatorName<Key>();
  std::fprintf(stderr,
               "[INFINI_OPS_TRACE_CALLS] {\"operator_name\": \"%.*s\", "
               "\"device_type\": \"%.*s\", \"implementation\": %zu}\n",
               static_cast<int>(operator_name.size()), operator_name.data(),
               static_cast<int>(device.size()), device.data(),
               config.implementation_index());
}

}  // namespace infini::ops::detail

template <>
struct std::hash<infini::ops::detail::CacheKey> {
  std::size_t operator()(const infini::ops::detail::CacheKey& key) const {
    return key.hash;
  }
};

template <>
struct std::equal_to<infini::ops::detail::CacheKey> {
  bool operator()(const infini::ops::detail::CacheKey& a,
                  const infini::ops::detail::CacheKey& b) const {
    if (a.scalar_hash != b.scalar_hash) return false;
    if (a.tensors.size() != b.tensors.size()) return false;
    std::equal_to<infini::ops::Tensor> eq;
    for (std::size_t i = 0; i < a.tensors.size(); ++i) {
      if (!eq(a.tensors[i], b.tensors[i])) return false;
    }
    return true;
  }
};

namespace infini::ops {

template <typename Key>
struct CacheKeyBuilder {
  template <typename... Args>
  detail::CacheKey operator()(const Config& config, const Args&... args) const {
    return detail::CacheKey::Build(config.implementation_index(), args...);
  }
};

namespace detail {

template <typename Key, typename... Args>
std::size_t ResolveImplementationIndex(const Config& config,
                                       Device::Type dev_type,
                                       const Args&... args);

template <typename Key, typename... Args>
std::size_t ResolveImplementationIndexOnline(const Handle& handle,
                                             const Config& config,
                                             Device::Type dev_type,
                                             const Args&... args);

}  // namespace detail

template <typename Key, Device::Type kDev>
struct ActiveImplementations;

class OperatorBase {
 public:
  virtual ~OperatorBase() = default;

  virtual std::size_t workspace_size_in_bytes() const { return 0; }

  void set_handle(const Handle& handle) { handle_ptr_ = handle.Clone(); }

  void set_config(const Config& config) { config_ptr_ = config.Clone(); }

  void set_stream(void* stream) { stream_ = stream; }

  void set_workspace(void* workspace) { workspace_ = workspace; }

  void set_workspace_size_in_bytes(std::size_t workspace_size_in_bytes) {
    workspace_size_in_bytes_ = workspace_size_in_bytes;
  }

 protected:
  std::unique_ptr<Handle> handle_ptr_;

  std::unique_ptr<Config> config_ptr_;

  void* stream_{nullptr};

  void* workspace_{nullptr};

  std::size_t workspace_size_in_bytes_{0};
};

template <typename Key, Device::Type device_type = Device::Type::kCount,
          std::size_t implementation_index = 0>
class Operator : public OperatorBase {
 public:
  // Invalidate the operator cache. Cached operators are destroyed on the next
  // `Call()` invocation. Intended for test isolation; production code should
  // never call this.
  static void clear_cache() {
    cache_generation_.fetch_add(1, std::memory_order_relaxed);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const Config& config,
                                        const Tensor tensor, Args&&... args) {
    const auto dev_type = tensor.device().type();
    if (!TuningManager::Instance().IsEnabled() ||
        !config.needs_implementation_resolution()) {
      return MakeWithDevice(config, dev_type, tensor,
                            std::forward<Args>(args)...);
    }

    auto resolved_config = config.Clone();
    resolved_config->set_implementation_index(
        detail::ResolveImplementationIndex<Key>(config, dev_type, tensor,
                                                args...));
    return MakeWithDevice(*resolved_config, dev_type, tensor,
                          std::forward<Args>(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const Tensor tensor, Args&&... args) {
    return Make(ImplicitConfig(tensor.device().type()), tensor,
                std::as_const(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const Config& config,
                                        const std::vector<Tensor> tensors,
                                        Args&&... args) {
    assert(!tensors.empty() && "operator tensor list input cannot be empty");

    const auto dev_type = tensors.front().device().type();
    if (!TuningManager::Instance().IsEnabled() ||
        !config.needs_implementation_resolution()) {
      return MakeWithDevice(config, dev_type, tensors,
                            std::forward<Args>(args)...);
    }

    auto resolved_config = config.Clone();
    resolved_config->set_implementation_index(
        detail::ResolveImplementationIndex<Key>(config, dev_type, tensors,
                                                args...));
    return MakeWithDevice(*resolved_config, dev_type, tensors,
                          std::forward<Args>(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const std::vector<Tensor> tensors,
                                        Args&&... args) {
    assert(!tensors.empty() && "operator tensor list input cannot be empty");

    return Make(ImplicitConfig(tensors.front().device().type()), tensors,
                std::as_const(args)...);
  }

  template <typename... Args>
  static void Call(const Handle& handle, const Config& config,
                   const Args&... args) {
    [[maybe_unused]] HostRangeScope host_range_operator_call{
        HostRangeLayer::kOperatorCall};

    static thread_local std::unordered_map<detail::CacheKey,
                                           std::unique_ptr<Operator>>
        cache;
    static thread_local std::size_t generation{0};

    const auto cache_generation =
        cache_generation_.load(std::memory_order_relaxed);
    if (generation != cache_generation) {
      cache.clear();
      generation = cache_generation;
    }

    std::unique_ptr<Config> resolved_config;
    const Config* effective_config = &config;
    if (TuningManager::Instance().IsEnabled() &&
        config.needs_implementation_resolution()) {
      const auto dev_type = detail::FirstDeviceType(args...);
      assert(dev_type != Device::Type::kCount &&
             "operator call requires at least one tensor argument");

      const auto resolved_implementation_index =
          detail::ResolveImplementationIndexOnline<Key>(handle, config,
                                                        dev_type, args...);
      resolved_config = config.Clone();
      resolved_config->set_implementation_index(resolved_implementation_index);
      effective_config = resolved_config.get();
    }

#if defined(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
    auto key = [&]() {
      HostRangeScope host_range_cache_key{HostRangeLayer::kCacheKey};
      return CacheKeyBuilder<Key>{}(*effective_config, args...);
    }();
    detail::TraceOperatorCall<Key>(key, *effective_config);

    auto it = [&]() {
      HostRangeScope host_range_cache_lookup{HostRangeLayer::kCacheLookup};
      return cache.find(key);
    }();

    if (it == cache.end()) {
      HostRangeScope host_range_cache_construct{
          HostRangeLayer::kCacheConstruct};
      auto new_op = Make(*effective_config, args...);
      it = cache.emplace(std::move(key), std::move(new_op)).first;
    }
#else
    auto key = CacheKeyBuilder<Key>{}(*effective_config, args...);
    detail::TraceOperatorCall<Key>(key, *effective_config);

    auto it{cache.find(key)};

    if (it == cache.end()) {
      it =
          cache.emplace(std::move(key), Make(*effective_config, args...)).first;
    }
#endif

    auto& op{it->second};

    [[maybe_unused]] HostRangeScope host_range_operator_invoke{
        HostRangeLayer::kOperatorInvoke};
    return (*op)(handle, args...);
  }

  template <typename... Args>
  static void Call(const Tensor tensor, const Args&... args) {
    return Call({}, ImplicitConfig(tensor.device().type()), tensor, args...);
  }

  template <
      typename TensorLike, typename... Args,
      typename std::enable_if_t<
          detail::HasMakeReturnValue<Key, TensorLike, Args...>::value, int> = 0>
  static auto Call(const TensorLike& tensor, const Args&... args) {
    return CallReturning(tensor, args...);
  }

  static std::vector<std::size_t> active_implementation_indices(
      Device::Type dev_type) {
    if (!detail::ListContains(dev_type, ActiveDevices<Key>{})) {
      return {};
    }

    std::vector<std::size_t> result;
    DispatchFunc<ActiveDevices<Key>>(
        dev_type,
        [&](auto device_tag) {
          constexpr Device::Type kDev = decltype(device_tag)::value;
          result = detail::ListToVector(
              typename ActiveImplementations<Key, kDev>::type{});
        },
        "Operator::active_implementation_indices");
    return result;
  }

  template <typename... Args>
  void operator()(const Handle& handle, const Args&... args) {
    set_handle(handle);
    set_stream(handle.stream());
    set_workspace(handle.workspace());
    set_workspace_size_in_bytes(handle.workspace_size_in_bytes());

    return operator()(args...);
  }

  template <typename... Args>
  void operator()(const Args&... args) const {
    return (*static_cast<const Key*>(this))(args...);
  }

 protected:
  static constexpr Device::Type device_type_{device_type};

  static constexpr std::size_t implementation_index_{implementation_index};

 private:
  template <auto first, auto... rest>
  static constexpr std::size_t FirstActiveImplementationIndex(
      List<first, rest...>) {
    return static_cast<std::size_t>(first);
  }

  static std::size_t FirstActiveImplementationIndex(List<>) {
    assert(false && "operator has no active implementation for this device");
    std::abort();
  }

  static std::size_t DefaultImplementationIndex(Device::Type dev_type) {
    std::size_t default_index{0};

    DispatchFunc<ActiveDevices<Key>>(
        dev_type,
        [&](auto device_tag) {
          constexpr Device::Type kDev = decltype(device_tag)::value;
          default_index = FirstActiveImplementationIndex(
              typename ActiveImplementations<Key, kDev>::type{});
        },
        "Operator::DefaultImplementationIndex");

    return default_index;
  }

  static Config DefaultConfig(Device::Type dev_type) {
    Config config;
    config.set_implementation_index(DefaultImplementationIndex(dev_type));

    return config;
  }

  static Config ImplicitConfig(Device::Type dev_type) {
    if (TuningManager::Instance().IsEnabled()) return Config{};
    return DefaultConfig(dev_type);
  }

  template <typename TensorLike, typename... Args>
  static auto CallReturning(const TensorLike& tensor, const Args&... args) {
    auto out = Key::MakeReturnValue(tensor, args...);
    Key::Call(detail::AsCallArg(tensor), detail::AsCallArg(args)...,
              detail::AsCallArg(out));
    return out;
  }

  template <typename... Args>
  static std::unique_ptr<Operator> MakeWithDevice(
      const Config& config, Device::Type dispatch_device_type, Args&&... args) {
    std::unique_ptr<Operator> op_ptr;
    auto cache_args = std::forward_as_tuple(args...);

    DispatchFunc<ActiveDevices<Key>>(
        dispatch_device_type,
        [&](auto device_tag) {
          constexpr Device::Type kDev = decltype(device_tag)::value;
          detail::DispatchImplementation(
              config.implementation_index(),
              [&](auto implementation_tag) {
                constexpr std::size_t kImplementationIndex =
                    decltype(implementation_tag)::value;
                if constexpr (std::is_constructible_v<
                                  Operator<Key, kDev, kImplementationIndex>,
                                  Args...>) {
                  std::apply(
                      [&](auto&... cached_args) {
                        op_ptr = std::make_unique<
                            Operator<Key, kDev, kImplementationIndex>>(
                            cached_args...);
                      },
                      cache_args);
                } else {
                  assert(false &&
                         "operator is not implemented for this device and "
                         "implementation index");
                }
              },
              "Operator::Make(implementation_index)",
              typename ActiveImplementations<Key, kDev>::type{});
        },
        "Operator::Make");

    op_ptr->set_config(config);

    return op_ptr;
  }

  static inline std::atomic<std::size_t> cache_generation_{0};
};

// Maximum number of implementation slots per (operator, device) pair.
// Increase this value when adding operators with more implementations.
constexpr std::size_t kMaxImplementations = 32;

// SFINAE-based implementation detection. A partial specialization
// `Operator<Key, kDev, N>` inherits from `Key` (the operator base class),
// while the unspecialized primary template inherits only from `OperatorBase`.
// `std::is_base_of` distinguishes the two at compile time, eliminating the
// need for manual `registry.h` files.
template <typename Key, Device::Type kDev, std::size_t N,
          bool = std::is_base_of_v<Key, Operator<Key, kDev, N>>>
struct ActiveImplementationsImpl {
  using type = List<>;
};

template <typename Key, Device::Type kDev, std::size_t N>
struct ActiveImplementationsImpl<Key, kDev, N, true> {
  using type = List<N>;
};

namespace detail {

template <typename Key, Device::Type kDev, typename Seq>
struct ActiveImplementationsHelper;

template <typename Key, Device::Type kDev, std::size_t... ns>
struct ActiveImplementationsHelper<Key, kDev, std::index_sequence<ns...>> {
  using type = typename Flatten<
      typename ActiveImplementationsImpl<Key, kDev, ns>::type...>::type;
};

}  // namespace detail

template <typename Key, Device::Type kDev>
struct ActiveImplementations {
  using type = typename detail::ActiveImplementationsHelper<
      Key, kDev, std::make_index_sequence<kMaxImplementations>>::type;
};

namespace detail {

template <typename Key, typename... Args>
std::size_t ResolveImplementationIndex(const Config& config,
                                       Device::Type dev_type,
                                       const Args&... args) {
  if (!config.needs_implementation_resolution()) {
    return config.implementation_index();
  }

  auto indices = Operator<Key>::active_implementation_indices(dev_type);
  if (indices.empty()) return config.implementation_index();

  auto signature = TuningSignature::Build(args...);
  constexpr auto op_name = OperatorName<Key>();
  auto tuned_index =
      TuningManager::Instance().Lookup(op_name, dev_type, signature);
  auto chosen = indices.front();

  if (tuned_index.has_value()) {
    bool is_valid = std::find(indices.begin(), indices.end(), *tuned_index) !=
                    indices.end();
    if (is_valid) {
      chosen = *tuned_index;
    } else {
      std::cerr << "[Tuning] Warning: tuned implementation " << *tuned_index
                << " for " << op_name << " on "
                << Device::StringFromType(dev_type)
                << " is not available (compiled indices:";
      for (auto idx : indices) std::cerr << " " << idx;
      std::cerr << "), falling back to " << chosen << std::endl;
    }
  }

  return chosen;
}

template <typename Key, typename... Args>
double BenchmarkImplementation(const Handle& handle, Device::Type dev_type,
                               std::size_t impl_index, const Args&... args) {
  Config fixed;
  fixed.set_implementation_index(impl_index);

  auto op = Operator<Key>::Make(fixed, args...);

  const auto& tuning = TuningManager::Instance();
  const int warmup = tuning.warmup_count();
  const int repeat = tuning.repeat_count();

  for (int i = 0; i < warmup; ++i) {
    (*op)(handle, args...);
  }
  SyncDevice(dev_type);

  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < repeat; ++i) {
    auto start = std::chrono::steady_clock::now();
    (*op)(handle, args...);
    SyncDevice(dev_type);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    best = std::min(best, elapsed);
  }
  return best;
}

template <typename Key, typename... Args>
std::size_t ResolveImplementationIndexOnline(const Handle& handle,
                                             const Config& config,
                                             Device::Type dev_type,
                                             const Args&... args) {
  if (!config.needs_implementation_resolution()) {
    return config.implementation_index();
  }

  auto& tuning = TuningManager::Instance();
  if (!tuning.IsEnabled()) {
    return ResolveImplementationIndex<Key>(config, dev_type, args...);
  }

  auto indices = Operator<Key>::active_implementation_indices(dev_type);
  if (indices.empty()) return config.implementation_index();

  auto signature = TuningSignature::Build(args...);
  constexpr auto op_name = OperatorName<Key>();
  auto tuned = tuning.Lookup(op_name, dev_type, signature);
  std::size_t chosen;

  if (tuned.has_value() &&
      std::find(indices.begin(), indices.end(), *tuned) != indices.end()) {
    chosen = *tuned;
  } else if (indices.size() == 1) {
    chosen = indices.front();
    tuning.Record(op_name, dev_type, signature, chosen);
    std::cout << "[Tuning] " << op_name << " on "
              << Device::StringFromType(dev_type)
              << ": single impl, chose index " << chosen << std::endl;
  } else {
    chosen = indices.front();
    double best_time = std::numeric_limits<double>::infinity();
    for (auto idx : indices) {
      double time =
          BenchmarkImplementation<Key>(handle, dev_type, idx, args...);
      if (time < best_time) {
        best_time = time;
        chosen = idx;
      }
    }
    tuning.Record(op_name, dev_type, signature, chosen);
    std::cout << "[Tuning] " << op_name << " on "
              << Device::StringFromType(dev_type) << ": benchmarked "
              << indices.size() << " impls, chose index " << chosen << " ("
              << best_time * 1e6 << " us)" << std::endl;
  }

  return chosen;
}

}  // namespace detail

}  // namespace infini::ops

#endif
