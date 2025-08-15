#ifndef __COS_METAX_H__
#define __COS_METAX_H__

namespace op::cos::metax {

typedef struct CosOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        return cos_(x);
    }
} CosOp;

} // namespace op::cos::metax

#endif // __COS_METAX_H__