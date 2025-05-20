#ifndef __INFINIOP_ELEMENTWISE_BANG_KERNEL_MLU__
#define __INFINIOP_ELEMENTWISE_BANG_KERNEL_MLU__

#include "../../devices/bang/bang_kernel_common.h"

using namespace device::bang::kernel;

// NRAM memory constraints
#define NRAM_MAX_SIZE 102400
constexpr size_t ALIGN_SIZE = 128; // Cambricon alignment requirement

/**
 * @brief Helper struct for computing input tensor indices considering broadcasting and striding.
 */
struct InputIndexer {
    size_t idx;
    size_t ndim;
    const bool *input_contiguous;
    const bool *input_broadcasted;
    const size_t *input_shapes;
    const ptrdiff_t *input_strides;
    const ptrdiff_t *output_strides;

    /**
     * @brief Computes memory offset for input tensor element.
     *
     * @param input_id    Input tensor ID.
     * @param element_idx Element index in output tensor.
     * @return size_t     Memory offset in input tensor.
     */
    __mlu_device__ size_t operator()(size_t input_id, size_t element_idx) const {
        size_t global_idx = idx + element_idx;
        return input_contiguous[input_id]
                 ? global_idx
                 : (input_broadcasted[input_id]
                        ? indexToReducedOffset(global_idx, ndim, output_strides, input_strides + input_id * ndim)
                        : indexToOffset(global_idx, ndim, input_shapes + input_id * ndim, input_strides + input_id * ndim));
    }
};

/**
 * @brief Computes output tensor index considering striding.
 *
 * @param idx            Linear index.
 * @param is_contiguous  Whether output is contiguous.
 * @param ndim           Number of dimensions.
 * @param shape          Output tensor shape.
 * @param strides        Output tensor strides.
 * @return size_t        Memory offset in output tensor.
 */
inline __mlu_device__ size_t
getOutputIndex(size_t idx,
               bool is_contiguous,
               size_t ndim,
               const size_t *shape,
               const ptrdiff_t *strides) {
    return is_contiguous ? idx : indexToOffset(idx, ndim, shape, strides);
}

/**
 * @brief Core elementwise operation implementation for BANG device.
 *
 * @tparam N              Number of input tensors.
 * @tparam Op            Operator functor type.
 * @tparam Tdata         Data type for inputs and output.
 * @tparam Args          Additional arguments for operator.
 *
 * @param typed_inputs    Array of typed input pointers.
 * @param output         Output tensor pointer.
 * @param nram_buf       NRAM buffer for temporary storage.
 * @param input_indexes  Precomputed input indexes.
 * @param output_index   Starting output index.
 * @param num_elements   Number of elements to process.
 * @param output_contiguous Whether output is contiguous.
 * @param input_contiguous Array indicating input contiguity.
 * @param ndim           Number of dimensions.
 * @param output_shape_gm Output shape in global memory.
 * @param output_strides_gm Output strides in global memory.
 * @param indexer        Input indexer helper.
 * @param start_idx      Starting index for this task.
 * @param args           Additional arguments for operator.
 */
template <size_t N, typename Op, typename Tdata, typename... Args>
__mlu_device__ void launchOp(
    Tdata **typed_inputs,               // global memory pointers
    Tdata *output,                      // global memory output
    Tdata *nram_buf,                    // NRAM buffer for inputs and output
    size_t *input_indexes,              // input indexes for each operand
    size_t output_index,                // output index
    size_t num_elements,                // number of elements to process
    bool output_contiguous,             // whether output is contiguous
    const bool *input_contiguous,       // whether each input is contiguous
    size_t ndim,                        // number of dimensions
    const size_t *output_shape_gm,      // output shape
    const ptrdiff_t *output_strides_gm, // output strides
    InputIndexer indexer,               // input indexer
    size_t start_idx,                   // starting index for this task
    Args... args) {

    static_assert(N == Op::num_inputs, "template N is not equal to Op::num_inputs!");

    // NRAM memory planning
    const size_t nram_usable = NRAM_MAX_SIZE - (ALIGN_SIZE * (N + 1));
    const size_t max_batch = nram_usable / ((N + 1) * sizeof(Tdata));

    size_t processed = 0;
    while (processed < num_elements) {
        size_t curr_batch = std::min(max_batch, num_elements - processed);

        // Align memory address
        Tdata *aligned_buf = reinterpret_cast<Tdata *>(
            (reinterpret_cast<size_t>(nram_buf) + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1));

        // 1. Copy input data to NRAM
        Tdata *input_buffers[N];
        for (size_t i = 0; i < N; ++i) {
            input_buffers[i] = aligned_buf + i * max_batch;

            if (input_contiguous[i]) {
                // Contiguous case - can use bulk copy
                __memcpy(input_buffers[i],
                         typed_inputs[i] + input_indexes[i] + processed,
                         curr_batch * sizeof(Tdata),
                         GDRAM2NRAM);
            } else {
                // Non-contiguous case - copy element by element
                for (size_t j = 0; j < curr_batch; j++) {
                    size_t element_offset = indexer(i, j + processed);
                    __memcpy(input_buffers[i] + j,
                             typed_inputs[i] + element_offset,
                             sizeof(Tdata),
                             GDRAM2NRAM);
                }
            }
        }
        __asm__ volatile("sync;");

        // 2. Execute operation
        Tdata *output_buffer = aligned_buf + N * max_batch;
        Op op;
        op(output_buffer, input_buffers[0], input_buffers[1], curr_batch, args...);
        __asm__ volatile("sync;");

        // 3. Write back results
        if (output_contiguous) {
            // Contiguous output - bulk copy
            __memcpy(output + output_index + processed,
                     output_buffer,
                     curr_batch * sizeof(Tdata),
                     NRAM2GDRAM);
        } else {
            // Non-contiguous output - copy element by element
            for (size_t j = 0; j < curr_batch; j++) {
                size_t out_offset = getOutputIndex(start_idx + j + processed,
                                                   output_contiguous,
                                                   ndim,
                                                   output_shape_gm,
                                                   output_strides_gm);
                __memcpy(output + out_offset,
                         output_buffer + j,
                         sizeof(Tdata),
                         NRAM2GDRAM);
            }
        }
        __asm__ volatile("sync;");

        processed += curr_batch;
    }
}

/**
 * @brief BANG kernel for elementwise operations.
 *
 * @tparam N        Number of input tensors.
 * @tparam Op       Operator functor type.
 * @tparam Tdata    Data type for inputs and output.
 * @tparam Args     Additional arguments for operator.
 *
 * @param output_size         Total output elements.
 * @param ndim                Number of dimensions.
 * @param output_contiguous   Whether output is contiguous.
 * @param input_contiguous_gm Input contiguity flags in global memory.
 * @param input_broadcasted_gm Input broadcast flags in global memory.
 * @param output_shape_gm     Output shape in global memory.
 * @param input_shapes_gm     Input shapes in global memory.
 * @param output_strides_gm   Output strides in global memory.
 * @param input_strides_gm    Input strides in global memory.
 * @param output             Output tensor pointer.
 * @param inputs             Array of input pointers.
 * @param args               Additional arguments for operator.
 */
template <size_t N, typename Op, typename Tdata, typename... Args>
__mlu_global__ void elementwiseKernel(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *input_contiguous_gm,
    const bool *input_broadcasted_gm,
    const size_t *output_shape_gm,
    const size_t *input_shapes_gm,
    const ptrdiff_t *output_strides_gm,
    const ptrdiff_t *input_strides_gm,
    Tdata *output,
    const void *const *inputs,
    Args... args) {

    // Cast input pointers to the correct type
    Tdata *typed_inputs[N];
    for (size_t i = 0; i < N; i++) {
        typed_inputs[i] = reinterpret_cast<Tdata *>(const_cast<void *>(inputs[i]));
    }

    // Calculate workload per task
    size_t elements_per_task = (output_size + taskDim - 1) / taskDim;
    size_t start_idx = taskId * elements_per_task;
    size_t end_idx = std::min(start_idx + elements_per_task, output_size);
    size_t num_elements = end_idx > start_idx ? end_idx - start_idx : 0;

    if (num_elements == 0) {
        return;
    }

    // Allocate NRAM buffer (shared by all inputs and output)
    __nram__ Tdata nram_buf[NRAM_MAX_SIZE / sizeof(Tdata)];

    // Get output index
    size_t output_index = getOutputIndex(start_idx, output_contiguous,
                                         ndim, output_shape_gm, output_strides_gm);

    // Create input indexer
    InputIndexer indexer{
        static_cast<size_t>(start_idx),
        ndim,
        input_contiguous_gm,
        input_broadcasted_gm,
        input_shapes_gm,
        input_strides_gm,
        output_strides_gm};

    // Get index offsets for each operand
    size_t input_indexes[N];
    for (size_t i = 0; i < N; i++) {
        input_indexes[i] = indexer(i, 0);
    }

    // Launch the operation with all required parameters
    launchOp<N, Op, Tdata>(typed_inputs, output, nram_buf, input_indexes,
                           output_index, num_elements, output_contiguous,
                           input_contiguous_gm, ndim, output_shape_gm,
                           output_strides_gm, indexer, start_idx, args...);

    __sync_all();
}

/**
 * @brief Determines optimal kernel launch configuration based on input size and hardware.
 *
 * @param output_size Total number of output elements.
 * @return cnrtDim3_t Optimal kernel launch dimensions.
 */
inline cnrtDim3_t getOptimalLaunchConfig() {
    cnrtDim3_t dim;

    // Get hardware information
    int cluster_num, core_num;
    cnrtDeviceGetAttribute(&cluster_num, cnrtAttrClusterCount, 0);
    cnrtDeviceGetAttribute(&core_num, cnrtAttrMcorePerCluster, 0);

    dim.x = core_num;
    dim.y = cluster_num;
    dim.z = 1;

    return dim;
}

/**
 * @brief Intermediate layer that determines optimal launch configuration before calling elementwiseKernel.
 *
 * @tparam N        Number of input tensors.
 * @tparam Op       Operator functor type.
 * @tparam Tdata    Data type for inputs and output.
 * @tparam Args     Additional arguments for operator.
 */
template <size_t N, typename Op, typename Tdata, typename... Args>
void launchElementwiseKernelWrapper(
    size_t output_size,
    size_t ndim,
    bool output_contiguous,
    const bool *input_contiguous,
    const bool *input_broadcasted,
    const size_t *output_shape,
    const size_t *input_shapes,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    Tdata *output,
    const void *const *inputs,
    cnrtQueue_t queue,
    Args... args) {

    // Determine optimal launch configuration
    cnrtDim3_t dim = getOptimalLaunchConfig();

    // Choose kernel type based on problem characteristics
    cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_BLOCK;
    if (output_size > 1024 * 1024 && output_contiguous) {
        // For large contiguous operations, use UNION type
        func_type = CNRT_FUNC_TYPE_UNION1;
    }

    // Launch the kernel with optimal configuration
    elementwiseKernel<N, Op, Tdata><<<dim, func_type, queue>>>(
        output_size, ndim, output_contiguous,
        input_contiguous, input_broadcasted,
        output_shape, input_shapes,
        output_strides, input_strides,
        output, inputs, args...);
}

/**
 * @brief Macro for implementing elementwise kernel launch.
 *
 * @param OpName Name of the operation.
 * @param Op     Operator functor type.
 */
#define LAUNCH_ELEMENTWISE_KERNEL_IMPL(OpName, Op)                      \
    template <typename Tdata, typename... Args>                         \
    void launch##OpName##Kernel(                                        \
        size_t output_size,                                             \
        size_t ndim,                                                    \
        bool output_contiguous,                                         \
        const void *input_contiguous,                                   \
        const void *input_broadcasted,                                  \
        const void *output_shape,                                       \
        const void *input_shapes,                                       \
        const void *output_strides,                                     \
        const void *input_strides,                                      \
        void *output,                                                   \
        const void *const *inputs,                                      \
        cnrtQueue_t queue,                                              \
        Args... args) {                                                 \
        launchElementwiseKernelWrapper<Op::num_inputs, Op, Tdata>(      \
            output_size, ndim, output_contiguous,                       \
            reinterpret_cast<const bool *>(input_contiguous),           \
            reinterpret_cast<const bool *>(input_broadcasted),          \
            reinterpret_cast<const size_t *>(output_shape),             \
            reinterpret_cast<const size_t *>(input_shapes),             \
            reinterpret_cast<const ptrdiff_t *>(output_strides),        \
            reinterpret_cast<const ptrdiff_t *>(input_strides),         \
            reinterpret_cast<Tdata *>(output), inputs, queue, args...); \
    }

/**
 * @brief Macro for instantiating elementwise kernel for specific types.
 *
 * @param OpName Name of the operation.
 * @param T      Data type.
 * @param ...    Additional template arguments.
 */
#define LAUNCH_ELEMENTWISE_KERNEL_INSTANTIATE(OpName, T, ...) \
    template void launch##OpName##Kernel<T, ##__VA_ARGS__>(   \
        size_t output_size,                                   \
        size_t ndim,                                          \
        bool output_contiguous,                               \
        const void *input_contiguous,                         \
        const void *input_broadcasted,                        \
        const void *output_shape,                             \
        const void *input_shapes,                             \
        const void *output_strides,                           \
        const void *input_strides,                            \
        void *output,                                         \
        const void *const *inputs,                            \
        cnrtQueue_t queue,                                    \
        ##__VA_ARGS__);

#endif
