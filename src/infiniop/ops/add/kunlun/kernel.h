#ifndef __ADD_KUNLUN_KERNEL_H__
#define __ADD_KUNLUN_KERNEL_H__

namespace op::add::kunlun {

typedef struct AddOp {
public:
    static constexpr int num_inputs = 2;
    template <typename T>
    inline __device__ T operator()(const T *inputs) const {
        T a = inputs[0];
        T b = inputs[1];
        return a + b;
    }
    // bfloat16 - cast to flloat
    inline __device__ bfloat16_t operator()(const bfloat16_t *inputs) const {
        float a_f = __bfloat162float(inputs[0]);
        float b_f = __bfloat162float(inputs[1]);
        return __float2bfloat16(a_f + b_f);
    }
    // int64_t - cast to int32_t
    inline __device__ int64_t operator()(const int64_t *inputs) const {
        int32_t a = static_cast<int32_t>(inputs[0]);
        int32_t b = static_cast<int32_t>(inputs[1]);
        return static_cast<int64_t>(a + b);
    }
} AddOp;
} // namespace op::add::kunlun

#endif // __ADD_KUNLUN_KERNEL_H__
