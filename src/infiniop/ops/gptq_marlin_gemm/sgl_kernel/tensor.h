/// \file tensor.h
/// \brief Tensor validation and symbolic matching utilities.
#pragma once
#include "utils.h"

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
#include "utils.cuh"
#endif

namespace host
{
  struct SymbolicSize;
  struct SymbolicDType;
  struct SymbolicDevice;

  namespace details
  {
    inline constexpr auto kAnyDeviceID = -1;
    inline constexpr auto kAnySize = static_cast<int64_t>(-1);
    inline constexpr auto kNullSize = static_cast<int64_t>(-1);
    inline constexpr auto kNullDType = static_cast<DLDataTypeCode>(18u);
    inline constexpr auto kNullDevice = static_cast<DLDeviceType>(-1);

    template <typename T>
    struct ArrayView
    {
      const T *data;
      size_t size;

      __host__ __device__ ArrayView() : data(nullptr), size(0) {}
      __host__ __device__ ArrayView(const T *d, size_t s) : data(d), size(s) {}

      template <size_t N>
      __host__ __device__ ArrayView(const std::array<T, N> &arr)
          : data(arr.data()), size(arr.size()) {}

      __host__ __device__ const T &operator[](size_t i) const { return data[i]; }
      __host__ __device__ bool empty() const { return size == 0; }
    };

    template <typename T>
    struct PrintAbleSpan
    {
      const T *data;
      size_t length;

      PrintAbleSpan(const T *p, size_t l) : data(p), length(l) {}
      size_t size() const { return length; }
      const T &operator[](size_t i) const { return data[i]; }
    };

    inline constexpr const char *kDeviceStringMap[] = {
        "",             // 0
        "cpu",          // 1
        "cuda",         // 2
        "cuda_host",    // 3
        "opencl",       // 4
        "vulkan",       // 5
        "metal",        // 6
        "vpi",          // 7
        "rocm",         // 8
        "rocm_host",    // 9
        "ext_dev",      // 10
        "cuda_managed", // 11
        "oneapi",       // 12
        "webgpu",       // 13
        "hexagon",      // 14
        "maia",         // 15
        "trn",          // 16
    };

    constexpr int kMaxDeviceType = 16;

    struct PrintableDevice
    {
      DLDevice device;
    };

    template <typename T>
    struct _dtype_trait;

    template <>
    struct _dtype_trait<int8_t>
    {
      static constexpr DLDataType value = {kDLInt, 8, 1};
    };
    template <>
    struct _dtype_trait<int16_t>
    {
      static constexpr DLDataType value = {kDLInt, 16, 1};
    };
    template <>
    struct _dtype_trait<int32_t>
    {
      static constexpr DLDataType value = {kDLInt, 32, 1};
    };
    template <>
    struct _dtype_trait<int64_t>
    {
      static constexpr DLDataType value = {kDLInt, 64, 1};
    };
    template <>
    struct _dtype_trait<uint8_t>
    {
      static constexpr DLDataType value = {kDLUInt, 8, 1};
    };
    template <>
    struct _dtype_trait<uint16_t>
    {
      static constexpr DLDataType value = {kDLUInt, 16, 1};
    };
    template <>
    struct _dtype_trait<uint32_t>
    {
      static constexpr DLDataType value = {kDLUInt, 32, 1};
    };
    template <>
    struct _dtype_trait<uint64_t>
    {
      static constexpr DLDataType value = {kDLUInt, 64, 1};
    };
    template <>
    struct _dtype_trait<float>
    {
      static constexpr DLDataType value = {kDLFloat, 32, 1};
    };
    template <>
    struct _dtype_trait<double>
    {
      static constexpr DLDataType value = {kDLFloat, 64, 1};
    };

#ifdef __CUDACC__
    template <>
    struct _dtype_trait<fp16_t>
    {
      static constexpr DLDataType value = {kDLFloat, 16, 1};
    };
    template <>
    struct _dtype_trait<bf16_t>
    {
      static constexpr DLDataType value = {kDLBfloat, 16, 1};
    };
    template <>
    struct _dtype_trait<fp8_e4m3_t>
    {
      static constexpr DLDataType value = {kDLFloat8_e4m3fn, 8, 1};
    };
#endif

    template <DLDeviceType Code>
    struct _device_trait
    {
      static constexpr DLDevice value = {Code, kAnyDeviceID};
    };

    template <typename... Ts>
    inline constexpr std::array<DLDataType, sizeof...(Ts)> kDTypeList = {
        _dtype_trait<Ts>::value...};

    template <DLDeviceType... Codes>
    inline constexpr std::array<DLDevice, sizeof...(Codes)> kDeviceList = {
        _device_trait<Codes>::value...};

  } // namespace details

  inline std::ostream &operator<<(std::ostream &os, DLDevice device)
  {
    int code = static_cast<int>(device.device_type);
    if (code < 1 || code > details::kMaxDeviceType)
      RuntimeCheck(false, "Unknown device: ", code);
    os << details::kDeviceStringMap[code];
    if (device.device_id != details::kAnyDeviceID && device.device_type != kDLCPU)
      os << ":" << device.device_id;
    return os;
  }

  inline std::ostream &operator<<(std::ostream &os, details::PrintableDevice pd)
  {
    return os << pd.device;
  }

  template <typename T>
  inline std::ostream &operator<<(std::ostream &os, const details::PrintAbleSpan<T> &span)
  {
    os << "[";
    for (size_t i = 0; i < span.size(); ++i)
    {
      if (i > 0)
        os << ", ";
      os << span[i];
    }
    os << "]";
    return os;
  }

  // ==============================================
  // SymbolicSize 完整定义
  // ==============================================
  struct SymbolicSize
  {
  public:
    explicit SymbolicSize(std::string_view ann = {})
        : m_value(details::kNullSize), m_ann(ann) {}

    SymbolicSize(const SymbolicSize &) = delete;
    SymbolicSize &operator=(const SymbolicSize &) = delete;

    std::string_view get_name() const { return m_ann; }
    bool has_value() const { return m_value != details::kNullSize; }

    void set_value(int64_t v)
    {
      RuntimeCheck(!has_value(), "Size already set");
      m_value = v;
    }

    std::optional<int64_t> get_value() const
    {
      return has_value() ? std::optional<int64_t>(m_value) : std::nullopt;
    }

    int64_t unwrap(DebugInfo info = {}) const
    {
      RuntimeCheck(info, has_value(), "Size not set");
      return m_value;
    }

    void verify(int64_t v, const char *prefix, int64_t dim)
    {
      if (has_value())
      {
        if (m_value != v) [[unlikely]]
        {
          Panic("Size mismatch for ", m_name_str(prefix, dim), ": expected ", m_value, " got ", v);
        }
      }
      else
      {
        set_value(v);
      }
    }

    std::string value_or_name(const char *prefix, int64_t dim) const
    {
      if (auto v = get_value())
        return std::to_string(*v);
      return m_name_str(prefix, dim);
    }

  private:
    std::string m_name_str(const char *prefix, int64_t dim) const
    {
      std::ostringstream os;
      os << prefix << '#' << dim;
      if (!m_ann.empty())
        os << "('" << m_ann << "')";
      return os.str();
    }

    int64_t m_value;
    std::string_view m_ann;
  };

  inline bool operator==(DLDevice a, DLDevice b)
  {
    return a.device_type == b.device_type && a.device_id == b.device_id;
  }

  // ==============================================
  // SymbolicDType 完整定义
  // ==============================================
  struct SymbolicDType
  {
  public:
    SymbolicDType() : m_value({details::kNullDType, 0, 0}) {}
    SymbolicDType(const SymbolicDType &) = delete;
    SymbolicDType &operator=(const SymbolicDType &) = delete;

    bool has_value() const { return m_value.code != details::kNullDType; }

    void set_value(DLDataType v)
    {
      RuntimeCheck(!has_value(), "DType already set");
      RuntimeCheck(m_check(v), "DType not allowed: ", v);
      m_value = v;
    }

    std::optional<DLDataType> get_value() const
    {
      return has_value() ? std::optional<DLDataType>(m_value) : std::nullopt;
    }

    DLDataType unwrap(DebugInfo info = {}) const
    {
      RuntimeCheck(info, has_value(), "DType not set");
      return m_value;
    }

    void set_options(details::ArrayView<const DLDataType> opts) { m_opts = opts; }

    template <typename... Ts>
    void set_options()
    {
      m_opts = details::ArrayView<const DLDataType>(details::kDTypeList<Ts...>.data(), details::kDTypeList<Ts...>.size());
    }

    void verify(DLDataType dtype)
    {
      if (has_value())
      {
        RuntimeCheck(m_value == dtype, "DType mismatch: expected ", m_value, " got ", dtype);
      }
      else
      {
        set_value(dtype);
      }
    }

    template <typename T>
    bool is_type() const
    {
      return m_value == details::_dtype_trait<T>::value;
    }

  private:
    bool m_check(DLDataType v) const
    {
      if (m_opts.empty())
        return true;
      for (size_t i = 0; i < m_opts.size; ++i)
        if (m_opts[i] == v)
          return true;
      return false;
    }

    details::ArrayView<const DLDataType> m_opts;
    DLDataType m_value;
  };

  // ==============================================
  // SymbolicDevice 完整定义
  // ==============================================
  struct SymbolicDevice
  {
  public:
    SymbolicDevice() : m_value({details::kNullDevice, details::kAnyDeviceID}) {}
    SymbolicDevice(const SymbolicDevice &) = delete;
    SymbolicDevice &operator=(const SymbolicDevice &) = delete;

    bool has_value() const { return m_value.device_type != details::kNullDevice; }

    void set_value(DLDevice v)
    {
      RuntimeCheck(!has_value(), "Device already set");
      RuntimeCheck(m_check(v), "Device not allowed: ", details::PrintableDevice{v});
      m_value = v;
    }

    std::optional<DLDevice> get_value() const
    {
      return has_value() ? std::optional<DLDevice>(m_value) : std::nullopt;
    }

    DLDevice unwrap(DebugInfo info = {}) const
    {
      RuntimeCheck(info, has_value(), "Device not set");
      return m_value;
    }

    void set_options(details::ArrayView<const DLDevice> opts) { m_opts = opts; }

    template <DLDeviceType... Codes>
    void set_options()
    {
      m_opts = details::ArrayView<const DLDevice>(details::kDeviceList<Codes...>.data(), details::kDeviceList<Codes...>.size());
    }

    void verify(DLDevice dev)
    {
      if (has_value())
      {
        RuntimeCheck(m_value == dev, "Device mismatch: expected ",
                     details::PrintableDevice{m_value}, " got ", details::PrintableDevice{dev});
      }
      else
      {
        set_value(dev);
      }
    }

  private:
    bool m_check(DLDevice v) const
    {
      if (m_opts.empty())
        return true;
      for (size_t i = 0; i < m_opts.size; ++i)
      {
        auto o = m_opts[i];
        if (o.device_type != v.device_type)
          continue;
        if (o.device_id == details::kAnyDeviceID || o.device_id == v.device_id)
          return true;
      }
      return false;
    }

    details::ArrayView<const DLDevice> m_opts;
    DLDevice m_value;
  };

  // ==============================================
  // BaseRef / Ref 类型（现在类型已完整定义）
  // ==============================================
  namespace details
  {
    template <typename T>
    struct BaseRef
    {
      BaseRef() : m_ref(&m_cache) {}
      explicit BaseRef(T &r) : m_ref(&r) {}
      BaseRef(const BaseRef &) = delete;
      BaseRef &operator=(const BaseRef &) = delete;

      T *operator->() const { return m_ref; }
      T &operator*() const { return *m_ref; }
      void rebind(T &r) { m_ref = &r; }

    private:
      T *m_ref;
      T m_cache;
    };

    struct SizeRef : public BaseRef<SymbolicSize>
    {
      using BaseRef::BaseRef;
      SizeRef(int64_t v);
    };

    struct DTypeRef : public BaseRef<SymbolicDType>
    {
      using BaseRef::BaseRef;
      DTypeRef(DLDataType);
      DTypeRef(std::initializer_list<DLDataType>);
      DTypeRef(ArrayView<const DLDataType>);
    };

    struct DeviceRef : public BaseRef<SymbolicDevice>
    {
      using BaseRef::BaseRef;
      DeviceRef(DLDevice);
      DeviceRef(std::initializer_list<DLDevice>);
      DeviceRef(ArrayView<const DLDevice>);
    };

    inline SizeRef::SizeRef(int64_t v)
    {
      if (v != kAnySize)
        (**this).set_value(v);
    }
    inline DTypeRef::DTypeRef(DLDataType v) { (**this).set_value(v); }
    inline DTypeRef::DTypeRef(std::initializer_list<DLDataType> l) : DTypeRef(ArrayView<const DLDataType>(l.begin(), l.size())) {}
    inline DTypeRef::DTypeRef(ArrayView<const DLDataType> v) { (**this).set_options(v); }
    inline DeviceRef::DeviceRef(DLDevice v) { (**this).set_value(v); }
    inline DeviceRef::DeviceRef(std::initializer_list<DLDevice> l) : DeviceRef(ArrayView<const DLDevice>(l.begin(), l.size())) {}
    inline DeviceRef::DeviceRef(ArrayView<const DLDevice> v) { (**this).set_options(v); }

  } // namespace details

  template <typename T>
  inline bool is_type(DLDataType dtype)
  {
    return dtype == details::_dtype_trait<T>::value;
  }

  // ==============================================
  // TensorMatcher
  // ==============================================
  struct TensorMatcher
  {
    using SizeRef = details::SizeRef;
    using DTypeRef = details::DTypeRef;
    using DeviceRef = details::DeviceRef;

    TensorMatcher(const TensorMatcher &) = delete;
    TensorMatcher &operator=(const TensorMatcher &) = delete;

    explicit TensorMatcher(std::initializer_list<SizeRef> s)
        : m_shape(s.begin(), s.size()), m_strides(nullptr, 0) {}

    TensorMatcher &&with_strides(std::initializer_list<SizeRef> s) &&
    {
      RuntimeCheck(m_strides.empty(), "Strides already set");
      RuntimeCheck(m_shape.size == s.size(), "Stride/shape size mismatch");
      m_strides = details::ArrayView<const SizeRef>(s.begin(), s.size());
      return std::move(*this);
    }

    template <typename... Ts>
    TensorMatcher &&with_dtype(DTypeRef &&d) &&
    {
      m_dtype.rebind(*d);
      m_dtype->template set_options<Ts...>();
      return std::move(*this);
    }

    template <typename... Ts>
    TensorMatcher &&with_dtype() &&
    {
      m_dtype->template set_options<Ts...>();
      return std::move(*this);
    }

    template <DLDeviceType... Codes>
    TensorMatcher &&with_device(DeviceRef &&d) &&
    {
      m_device.rebind(*d);
      m_device->template set_options<Codes...>();
      return std::move(*this);
    }

    template <DLDeviceType... Codes>
    TensorMatcher &&with_device() &&
    {
      m_device->template set_options<Codes...>();
      return std::move(*this);
    }

    const TensorMatcher &&verify(tvm::ffi::TensorView, DebugInfo = {}) const &&;

  private:
    static void s_print_tensor(std::ostringstream &, tvm::ffi::TensorView);
    void m_verify_impl(tvm::ffi::TensorView) const;

    details::ArrayView<const SizeRef> m_shape;
    details::ArrayView<const SizeRef> m_strides;
    DTypeRef m_dtype;
    DeviceRef m_device;
  };

  inline void TensorMatcher::s_print_tensor(std::ostringstream &os, tvm::ffi::TensorView v)
  {
    os << "Tensor<";
    size_t d = 0;
    for (int64_t s : v.shape())
    {
      if (d++)
        os << ", ";
      os << s;
    }
    os << ">[strides=<";
    d = 0;
    for (int64_t s : v.strides())
    {
      if (d++)
        os << ", ";
      os << s;
    }
    os << ">, dtype=" << v.dtype();
    os << ", device=" << details::PrintableDevice{v.device()} << "]";
  }

  inline const TensorMatcher &&TensorMatcher::verify(tvm::ffi::TensorView v, DebugInfo info) const &&
  {
    try
    {
      m_verify_impl(v);
    }
    catch (PanicError &e)
    {
      std::ostringstream os;
      os << "Tensor match failed: ";
      s_print_tensor(os, v);
      os << " @ " << info.file_name() << ":" << info.line() << "\n- cause: " << e.root_cause();
      throw PanicError(os.str());
    }
    return std::move(*this);
  }

  inline void TensorMatcher::m_verify_impl(tvm::ffi::TensorView v) const
  {
    size_t dim = static_cast<size_t>(v.dim());
    RuntimeCheck(dim == m_shape.size, "Dim mismatch: expected ", m_shape.size, " got ", dim);

    for (size_t i = 0; i < dim; ++i)
      m_shape[i]->verify(v.size(i), "shape", (int64_t)i);

    if (!m_strides.empty())
    {
      for (size_t i = 0; i < dim; ++i)
      {
        if (v.size(i) != 1 || !m_strides[i]->has_value())
          m_strides[i]->verify(v.stride(i), "stride", (int64_t)i);
      }
    }
    else
    {
      RuntimeCheck(v.is_contiguous(), "Tensor not contiguous");
    }

    m_dtype->verify(v.dtype());
    m_device->verify(v.device());
  }

} // namespace host
