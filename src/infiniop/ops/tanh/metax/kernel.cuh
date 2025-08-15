#ifndef __TANH_METAX_H__
#define __TANH_METAX_H__

namespace op::tanh::metax {

typedef struct TanhOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        return tanh_(x);
    }
} TanhOp;

} // namespace op::tanh::metax

#endif // __TANH_METAX_H__