namespace op::cast::cuda {

typedef struct CastOp {
public:
    static constexpr size_t num_inputs = 1;

private:
    template <typename T_src, typename T_dst>
    __device__ __forceinline__ T_dst cast_impl(const T_src &x) const {
        if constexpr (std::is_same_v<T_src, T_dst>) {
            return x;
        } else if constexpr (std::is_same_v<T_src, half>) {
            // From half
            if constexpr (std::is_same_v<T_dst, float>) {
                return __half2float(x);
            } else if constexpr (std::is_same_v<T_dst, double>) {
                return static_cast<double>(__half2float(x));
            } else if constexpr (std::is_same_v<T_dst, cuda_bfloat16>) {
                return __float2bfloat16(__half2float(x));
            } else if constexpr (std::is_integral_v<T_dst>) {
                return static_cast<T_dst>(__half2float(x));
            } else {
                return static_cast<T_dst>(__half2float(x));
            }
        } else if constexpr (std::is_same_v<T_src, cuda_bfloat16>) {
            // From bfloat16
            if constexpr (std::is_same_v<T_dst, float>) {
                return __bfloat162float(x);
            } else if constexpr (std::is_same_v<T_dst, double>) {
                return static_cast<double>(__bfloat162float(x));
            } else if constexpr (std::is_same_v<T_dst, half>) {
                return __float2half(__bfloat162float(x));
            } else if constexpr (std::is_integral_v<T_dst>) {
                return static_cast<T_dst>(__bfloat162float(x));
            } else {
                return static_cast<T_dst>(__bfloat162float(x));
            }
        } else if constexpr (std::is_same_v<T_dst, half>) {
            // To half
            if constexpr (std::is_same_v<T_src, float>) {
                return __float2half(x);
            } else if constexpr (std::is_same_v<T_src, double>) {
                return __float2half(static_cast<float>(x));
            } else {
                return __float2half(static_cast<float>(x));
            }
        } else if constexpr (std::is_same_v<T_dst, cuda_bfloat16>) {
            // To bfloat16
            if constexpr (std::is_same_v<T_src, float>) {
                return __float2bfloat16(x);
            } else if constexpr (std::is_same_v<T_src, double>) {
                return __float2bfloat16(static_cast<float>(x));
            } else {
                return __float2bfloat16(static_cast<float>(x));
            }
        } else if constexpr (std::is_same_v<T_src, half2>) {
            // Handle half2 special case
            if constexpr (std::is_same_v<T_dst, float>) {
                return __half2float(__low2half(x));
            } else {
                return static_cast<T_dst>(__half2float(__low2half(x)));
            }
        } else {
            // Direct cast for other cases
            return static_cast<T_dst>(x);
        }
    }

public:
    template <typename T_dst, typename T_src>
    __device__ __forceinline__ T_dst operator()(const T_src &x) const {
        return cast_impl<T_src, T_dst>(x);
    }

} CastOp;

} // namespace op::cast::cuda
