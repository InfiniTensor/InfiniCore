#ifndef __TANH_CUDA_H__
#define __TANH_CUDA_H__

namespace op::tanh::cuda {

// 预计算的tanh查找表，用于快速近似
__device__ __constant__ float tanh_lut[257] = {
    -0.999329f, -0.999286f, -0.99924f, -0.999191f, -0.999139f, -0.999083f, -0.999024f, -0.998961f,
    -0.998894f, -0.998823f, -0.998747f, -0.998667f, -0.998581f, -0.998489f, -0.998392f, -0.998288f,
    -0.998178f, -0.998061f, -0.997936f, -0.997803f, -0.997661f, -0.99751f, -0.99735f, -0.997179f,
    -0.996998f, -0.996804f, -0.996599f, -0.99638f, -0.996147f, -0.995898f, -0.995635f, -0.995354f,
    -0.995055f, -0.994737f, -0.994398f, -0.994038f, -0.993655f, -0.993247f, -0.992813f, -0.992351f,
    -0.99186f, -0.991337f, -0.990781f, -0.990189f, -0.98956f, -0.98889f, -0.988178f, -0.98742f,
    -0.986614f, -0.985757f, -0.984846f, -0.983876f, -0.982845f, -0.981749f, -0.980583f, -0.979344f,
    -0.978026f, -0.976626f, -0.975137f, -0.973554f, -0.971873f, -0.970086f, -0.968187f, -0.96617f,
    -0.964028f, -0.961752f, -0.959335f, -0.956769f, -0.954045f, -0.951154f, -0.948085f, -0.944829f,
    -0.941376f, -0.937712f, -0.933828f, -0.92971f, -0.925346f, -0.920722f, -0.915825f, -0.910638f,
    -0.905148f, -0.899339f, -0.893193f, -0.886695f, -0.879827f, -0.87257f, -0.864907f, -0.856818f,
    -0.848284f, -0.839285f, -0.829802f, -0.819814f, -0.809301f, -0.798243f, -0.786619f, -0.774409f,
    -0.761594f, -0.748154f, -0.734071f, -0.719328f, -0.703906f, -0.68779f, -0.670967f, -0.653424f,
    -0.635149f, -0.616134f, -0.596374f, -0.575862f, -0.5546f, -0.532587f, -0.50983f, -0.486336f,
    -0.462117f, -0.437189f, -0.41157f, -0.385284f, -0.358357f, -0.330821f, -0.30271f, -0.274062f,
    -0.244919f, -0.215326f, -0.185333f, -0.154991f, -0.124353f, -0.0934763f, -0.0624187f, -0.0312398f,
    0.0f, 0.0312398f, 0.0624187f, 0.0934763f, 0.124353f, 0.154991f, 0.185333f, 0.215326f,
    0.244919f, 0.274062f, 0.30271f, 0.330821f, 0.358357f, 0.385284f, 0.41157f, 0.437189f,
    0.462117f, 0.486336f, 0.50983f, 0.532587f, 0.5546f, 0.575862f, 0.596374f, 0.616134f,
    0.635149f, 0.653424f, 0.670967f, 0.68779f, 0.703906f, 0.719328f, 0.734071f, 0.748154f,
    0.761594f, 0.774409f, 0.786619f, 0.798243f, 0.809301f, 0.819814f, 0.829802f, 0.839285f,
    0.848284f, 0.856818f, 0.864907f, 0.87257f, 0.879827f, 0.886695f, 0.893193f, 0.899339f,
    0.905148f, 0.910638f, 0.915825f, 0.920722f, 0.925346f, 0.92971f, 0.933828f, 0.937712f,
    0.941376f, 0.944829f, 0.948085f, 0.951154f, 0.954045f, 0.956769f, 0.959335f, 0.961752f,
    0.964028f, 0.96617f, 0.968187f, 0.970086f, 0.971873f, 0.973554f, 0.975137f, 0.976626f,
    0.978026f, 0.979344f, 0.980583f, 0.981749f, 0.982845f, 0.983876f, 0.984846f, 0.985757f,
    0.986614f, 0.98742f, 0.988178f, 0.98889f, 0.98956f, 0.990189f, 0.990781f, 0.991337f,
    0.99186f, 0.992351f, 0.992813f, 0.993247f, 0.993655f, 0.994038f, 0.994398f, 0.994737f,
    0.995055f, 0.995354f, 0.995635f, 0.995898f, 0.996147f, 0.99638f, 0.996599f, 0.996804f,
    0.996998f, 0.997179f, 0.99735f, 0.99751f, 0.997661f, 0.997803f, 0.997936f, 0.998061f,
    0.998178f, 0.998288f, 0.998392f, 0.998489f, 0.998581f, 0.998667f, 0.998747f, 0.998823f,
    0.998894f, 0.998961f, 0.999024f, 0.999083f, 0.999139f, 0.999191f, 0.99924f, 0.999286f,
    0.999329f
};


// 查表法实现（高性能版本）- 使用预计算的查找表
template<typename T>
__device__ __forceinline__ T fast_tanh_lut(T x) {
    constexpr int LUT_SIZE = 256;
    constexpr float RANGE = 4.0f; // [-4, 4]
    
    float fx;
    if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        fx = __bfloat162float(x);
    } else {
        fx = static_cast<float>(x);
    }
    
    // 饱和处理
    if (fx >= RANGE) {
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16(1.0f);
        } else {
            return static_cast<T>(1.0f);
        }
    }
    if (fx <= -RANGE) {
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16(-1.0f);
        } else {
            return static_cast<T>(-1.0f);
        }
    }
    
    // 映射到查找表索引
    float normalized = (fx + RANGE) / (2.0f * RANGE);
    float index_f = normalized * LUT_SIZE;
    int index = static_cast<int>(index_f);
    float frac = index_f - index;
    
    // 边界检查
    if (index >= LUT_SIZE) index = LUT_SIZE - 1;
    if (index < 0) index = 0;
    
    // 使用预计算的查找表进行线性插值
    float y1 = tanh_lut[index];
    float y2 = (index + 1 < 257) ? tanh_lut[index + 1] : 1.0f;
    
    float result = y1 + frac * (y2 - y1);
    
    if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __float2bfloat16(result);
    } else {
        return static_cast<T>(result);
    }
}

typedef struct TanhOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2tanh(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return __float2half(tanhf(__half2float(x)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 对于bfloat16，使用查表法以获得最佳性能
            return fast_tanh_lut(x);
        } else if constexpr (std::is_same_v<T, float>) {
            // 对于float，使用CUDA内置的tanhf函数确保精度
            return tanhf(x);
        } else {
            return ::tanh(x);
        }
    }
} TanhOp;

// 高精度版本（保持与标准库一致）
typedef struct TanhOpHighPrecision {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2tanh(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return __float2half(tanhf(__half2float(x)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 高精度版本：显式转换并使用double作为中间计算类型
            float x_float = __bfloat162float(x);
            double x_double = static_cast<double>(x_float);
            double result = ::tanh(x_double);
            return __float2bfloat16(static_cast<float>(result));
        } else if constexpr (std::is_same_v<T, float>) {
            return tanhf(x);
        } else {
            return ::tanh(x);
        }
    }
} TanhOpHighPrecision;

} // namespace op::tanh::cuda

#endif // __TANH_CUDA_H__