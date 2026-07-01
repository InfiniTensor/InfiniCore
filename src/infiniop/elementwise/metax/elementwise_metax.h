#ifndef __INFINIOP_ELEMENTWISE_METAX_H__
#define __INFINIOP_ELEMENTWISE_METAX_H__

#include "../../../utils.h"
#include "../../devices/metax/metax_common.h"
#include "../../devices/metax/metax_kernel_common.h"
#include "elementwise_metax_api.h"
#include <cstdint>

namespace op::elementwise::metax {
template <typename T>
__device__ __forceinline__ const T *typedInputPtr(const void *ptr) {
    return reinterpret_cast<const T *>(ptr);
}

// Generic aligned N-element pack used for vectorized load/store.
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T val[N];
};

// Per-dtype vectorization info. Only floating-point types are enabled.
// Integer/bool/fp8 types keep using the scalar fallback automatically.
template <typename T>
struct VecInfo {
    static constexpr bool enabled = false;
};

template <>
struct VecInfo<float> {
    static constexpr bool enabled = true;
    static constexpr int pack_size = 4;
    using Type = Pack<float, pack_size>;
};

template <>
struct VecInfo<half> {
    static constexpr bool enabled = true;
    static constexpr int pack_size = 8;
    using Type = Pack<half, pack_size>;
};

template <>
struct VecInfo<cuda_bfloat16> {
    static constexpr bool enabled = true;
    static constexpr int pack_size = 8;
    using Type = Pack<cuda_bfloat16, pack_size>;
};

template <>
struct VecInfo<double> {
    static constexpr bool enabled = true;
    static constexpr int pack_size = 2;
    using Type = Pack<double, pack_size>;
};

template <typename Tdata, typename VecT, size_t N>
__device__ __forceinline__ void loadInputVectors(VecT *in_vecs,
                                                 const Tdata *const *typed_inputs,
                                                 size_t base) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
        in_vecs[i] = *reinterpret_cast<const VecT *>(typed_inputs[i] + base);
    }
}

__device__ __forceinline__ size_t getOutputIndex(size_t idx, bool is_contiguous, size_t ndim,
                                                 const size_t *shape, const ptrdiff_t *strides) {
    return is_contiguous ? idx : device::metax::indexToOffset(idx, ndim, shape, strides);
}

template <typename Tdata, size_t N>
bool canUseVecPath(const op::elementwise::ElementwiseInfo &info,
                   const void *output,
                   const std::vector<const void *> &inputs) {
    if constexpr (!VecInfo<Tdata>::enabled) {
        return false;
    }
    if (!info.isOutputContiguous()) {
        return false;
    }
    const bool *contiguous = info.getInputContiguous();
    const bool *broadcasted = info.getInputBroadcasted();
    for (size_t i = 0; i < N; ++i) {
        if (!contiguous[i] || broadcasted[i]) {
            return false;
        }
    }
    // Require 16-byte alignment of output and every input pointer.
    constexpr std::uintptr_t mask = 0xF;
    if ((reinterpret_cast<std::uintptr_t>(output) & mask) != 0) {
        return false;
    }
    for (const void *p : inputs) {
        if ((reinterpret_cast<std::uintptr_t>(p) & mask) != 0) {
            return false;
        }
    }
    return true;
}

struct InputIndexer {
    size_t idx;
    size_t ndim;
    const bool *input_contiguous;
    const bool *input_broadcasted;
    const size_t *input_shapes;
    const ptrdiff_t *input_strides;
    const ptrdiff_t *output_strides;

    __device__ __forceinline__ size_t operator()(size_t input_id) const {
        return input_contiguous[input_id]
                 ? idx
                 : device::metax::indexToOffset(idx, ndim, input_shapes + input_id * ndim, input_strides + input_id * ndim);
    }
};

template <typename F, size_t... Is>
__device__ __forceinline__ void unpackInputsAndApply(F &&f, std::index_sequence<Is...>) {
    f(std::integral_constant<size_t, Is>{}...);
}

template <size_t N, typename Op, typename Tdata, typename VecT, int V, typename... Args>
INFINIOP_METAX_KERNEL elementwiseVecKernel(
    size_t output_size,
    Tdata *__restrict__ output,
    const void *const *__restrict__ inputs,
    Args... args) {

    const Tdata *const *typed_inputs = reinterpret_cast<const Tdata *const *>(inputs);
    const size_t num_packs = output_size / V;
    const size_t tail_start = num_packs * V;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // Vectorized main loop over 16-byte packs
    for (size_t pack_idx = tid; pack_idx < num_packs; pack_idx += stride) {
        const size_t base = pack_idx * V;
        VecT in_vecs[N];
        loadInputVectors<Tdata, VecT, N>(in_vecs, typed_inputs, base);

        VecT out_vec;
#pragma unroll
        for (int k = 0; k < V; ++k) {
            unpackInputsAndApply(
                [&](auto... Is) {
                    out_vec.val[k] = Op{}(in_vecs[Is.value].val[k]..., std::forward<Args>(args)...);
                },
                std::make_index_sequence<N>{});
        }
        *reinterpret_cast<VecT *>(output + base) = out_vec;
    }

    // Scalar tail for remaining elements
    for (size_t idx = tail_start + tid; idx < output_size; idx += stride) {
        unpackInputsAndApply(
            [&](auto... Is) {
                output[idx] = Op{}(typed_inputs[Is.value][idx]..., std::forward<Args>(args)...);
            },
            std::make_index_sequence<N>{});
    }
}

template <size_t N, typename Op, typename Tdata, typename... Args>
INFINIOP_METAX_KERNEL elementwiseKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    Tdata *output,
    const void *const *inputs,
    size_t offset,
    Args... args) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < output_size) {
        const Tdata *const *typed_inputs = reinterpret_cast<const Tdata *const *>(inputs);
        size_t out_idx = getOutputIndex(idx, output_contiguous, ndim, output_shape, output_strides);
        InputIndexer indexer{idx, ndim, input_contiguous, input_broadcasted, input_shapes, input_strides, output_strides};

        unpackInputsAndApply(
            [&](auto... Is) {
                output[out_idx] = Op{}(typed_inputs[Is.value][indexer(Is.value)]..., std::forward<Args>(args)...);
            },
            std::make_index_sequence<N>{});
    }
}

template <typename Op, typename Tout, typename... Tin>
INFINIOP_METAX_KERNEL elementwiseKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *__restrict__ input_contiguous,
    const bool *__restrict__ input_broadcasted,
    const size_t *__restrict__ output_shape,
    const size_t *__restrict__ input_shapes,
    const ptrdiff_t *__restrict__ output_strides,
    const ptrdiff_t *__restrict__ input_strides,
    Tout *output,
    const void *const *__restrict__ inputs,
    size_t offset) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < output_size) {
        size_t out_idx = getOutputIndex(idx, output_contiguous, ndim, output_shape, output_strides);
        InputIndexer indexer{idx, ndim, input_contiguous, input_broadcasted, input_shapes, input_strides, output_strides};

        unpackInputsAndApply(
            [&](auto... Is) {
                output[out_idx] = Op{}.template operator()<Tout, Tin...>(
                    (typedInputPtr<Tin>(inputs[Is.value])[indexer(Is.value)])...);
            },
            std::index_sequence_for<Tin...>{});
    }
}

struct DeviceImpl::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;

    Opaque(const std::shared_ptr<device::metax::Handle::Internal> &internal)
        : internal(internal) {}

    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 hcStream_t stream,
                                 Args &&...args) {
        if constexpr (VecInfo<Tdata>::enabled) {
            if (canUseVecPath<Tdata, N>(info, output, inputs)) {
                return launchElementwiseVecKernel<BLOCK_SIZE, N, Op, Tdata,
                                                  typename VecInfo<Tdata>::Type,
                                                  VecInfo<Tdata>::pack_size>(
                    info, workspace,
                    reinterpret_cast<Tdata *>(output), inputs, stream,
                    std::forward<Args>(args)...);
            }
        }
        return launchElementwiseKernel<BLOCK_SIZE, N>(
            info, workspace,
            reinterpret_cast<Tdata *>(output), inputs,
            elementwiseKernel<N, Op, Tdata, Args...>,
            stream,
            std::forward<Args>(args)...);
    }

    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tout, typename... Tin, typename... Args,
              std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 hcStream_t stream,
                                 Args &&...args) {
        return launchElementwiseKernel<BLOCK_SIZE, N>(
            info, workspace,
            reinterpret_cast<Tout *>(output), inputs,
            elementwiseKernel<Op, Tout, Tin...>,
            stream);
    }

private:
    template <uint32_t BLOCK_SIZE, size_t N, typename Op, typename Tdata, typename VecT, int V, typename... Args>
    infiniStatus_t launchElementwiseVecKernel(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        Tdata *output,
        const std::vector<const void *> &inputs,
        hcStream_t stream,
        Args &&...args) {

        const auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        CHECK_METAX(hcMemcpyAsync(workspace, inputs.data(), N * sizeof(*inputs.data()),
                                  hcMemcpyHostToDevice, stream));
        const void **d_inputs_arr = reinterpret_cast<const void **>(workspace);

        dim3 blockDims(std::min(BLOCK_SIZE, static_cast<uint32_t>(internal->maxThreadsPerBlock())));
        const size_t num_packs = output_size / V;
        const size_t tail_size = output_size - num_packs * V;
        const size_t grid_work = num_packs > 0 ? num_packs : tail_size;
        dim3 gridDims(std::min(static_cast<uint32_t>(CEIL_DIV(grid_work, blockDims.x)),
                               static_cast<uint32_t>(internal->gridSizeX())));
        if (gridDims.x == 0) {
            gridDims.x = 1;
        }

        elementwiseVecKernel<N, Op, Tdata, VecT, V, Args...>
            <<<gridDims, blockDims, 0, stream>>>(
                output_size, output, d_inputs_arr, std::forward<Args>(args)...);

        return INFINI_STATUS_SUCCESS;
    }

    template <size_t N>
    infiniStatus_t infoToDevice(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        const void *const *h_inputs_arr,
        const void **&d_inputs_arr,
        const bool *&d_input_contiguous,
        const bool *&d_input_broadcasted,
        const size_t *&d_output_shape,
        const ptrdiff_t *&d_output_strides,
        const size_t *&d_input_shapes,
        const ptrdiff_t *&d_input_strides,
        hcStream_t stream) const {

        constexpr auto input_size = N;
        const auto ndim = info.getNdim();
        constexpr auto input_arr_size = N * sizeof(*h_inputs_arr);
        const int8_t *info_meta_start = info.getMetaStart();
        const int8_t *d_meta_start = reinterpret_cast<int8_t *>(workspace) + input_arr_size;

        // copy the input pointer array and meta to device
        CHECK_METAX(hcMemcpyAsync(workspace, h_inputs_arr, input_arr_size, hcMemcpyHostToDevice, stream));
        CHECK_METAX(hcMemcpyAsync((void *)d_meta_start, info_meta_start, info.getMetaMemSize(), hcMemcpyHostToDevice, stream));

        // offset/assign the pointers
        d_inputs_arr = reinterpret_cast<const void **>(workspace);
        d_output_shape = reinterpret_cast<const size_t *>(d_meta_start);
        d_output_strides = reinterpret_cast<const ptrdiff_t *>(d_output_shape + ndim);
        d_input_shapes = reinterpret_cast<const size_t *>(d_output_strides + ndim);
        d_input_strides = reinterpret_cast<const ptrdiff_t *>(d_input_shapes + input_size * ndim);
        d_input_contiguous = reinterpret_cast<const bool *>(d_input_strides + input_size * ndim);
        d_input_broadcasted = reinterpret_cast<const bool *>(d_input_contiguous + input_size);

        return INFINI_STATUS_SUCCESS;
    }

    template <uint32_t BLOCK_SIZE, size_t N, typename KernelFunc, typename Tout, typename... Args>
    infiniStatus_t launchElementwiseKernel(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        Tout *output,
        const std::vector<const void *> &inputs,
        KernelFunc kernel_func,
        hcStream_t stream,
        Args &&...args) {

        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        // Device pointers
        const void **d_inputs_arr = nullptr;
        const bool *d_input_contiguous = nullptr;
        const bool *d_input_broadcasted = nullptr;
        const size_t *d_output_shape = nullptr;
        const ptrdiff_t *d_output_strides = nullptr;
        const size_t *d_input_shapes = nullptr;
        const ptrdiff_t *d_input_strides = nullptr;

        CHECK_STATUS(infoToDevice<N>(info, workspace, inputs.data(), d_inputs_arr,
                                     d_input_contiguous, d_input_broadcasted,
                                     d_output_shape, d_output_strides,
                                     d_input_shapes, d_input_strides, stream));

        dim3 blockDims(std::min(BLOCK_SIZE, static_cast<uint32_t>(internal->maxThreadsPerBlock())));
        dim3 gridDims(std::min(uint32_t(CEIL_DIV(output_size, blockDims.x)), static_cast<uint32_t>(internal->gridSizeX())));
        size_t step = gridDims.x * blockDims.x;

        for (size_t i = 0; i < output_size; i += step) {
            kernel_func<<<gridDims, blockDims, 0, stream>>>(
                output_size, info.getNdim(), info.isOutputContiguous(),
                d_input_contiguous, d_input_broadcasted,
                d_output_shape, d_input_shapes,
                d_output_strides, d_input_strides,
                output, reinterpret_cast<const void **>(d_inputs_arr),
                i, std::forward<Args>(args)...);
        }

        return INFINI_STATUS_SUCCESS;
    }
};

template <typename... Args>
utils::Result<DeviceImpl *> DeviceImpl::create(Args &&...args) {
    auto opaque = std::make_shared<Opaque>(std::forward<Args>(args)...);
    return utils::Result<DeviceImpl *>(new DeviceImpl(opaque));
}

/* Invoke elementwise operation for different input types */
template <uint32_t BLOCK_SIZE, typename Op, typename Tout, typename... Tin, typename... Args,
          std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int>>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    static_assert(sizeof...(Tin) == N, "Input type count mismatch");
    return _opaque->calculateImpl<BLOCK_SIZE, N, Op, Tout, Tin...>(
        info, workspace, output, inputs,
        reinterpret_cast<hcStream_t>(stream),
        std::forward<Args>(args)...);
}

/* Invoke elementwise operation when all inputs have the same dtype */
template <uint32_t BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    return _opaque->calculateImpl<BLOCK_SIZE, N, Op, Tdata>(
        info, workspace, output, inputs,
        reinterpret_cast<hcStream_t>(stream),
        std::forward<Args>(args)...);
}

} // namespace op::elementwise::metax

#endif
