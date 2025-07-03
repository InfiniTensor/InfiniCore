#ifndef __INFINIOP_BANG_KERNEL_COMMON_H__
#define __INFINIOP_BANG_KERNEL_COMMON_H__

// Include Cambricon CNNL and CNRT headers for MLU (Machine Learning Unit) specific functions
#include "cnnl.h"
#include "cnrt.h"

namespace device::bang::kernel {

/**
 * @brief Converts a flattened index to a reduced offset considering broadcasting.
 *
 * This function is used when dealing with broadcasted tensors where the input
 * has been broadcast to match the output shape. It calculates the offset in
 * the original (non-broadcasted) tensor.
 *
 * @param flat_index The flattened index in the output tensor
 * @param ndim Number of dimensions
 * @param broadcasted_strides Strides of the broadcasted tensor
 * @param target_strides Strides of the original (non-broadcasted) tensor
 * @return size_t Offset in the original tensor's memory
 */
inline __mlu_device__ size_t indexToReducedOffset(
    size_t flat_index,
    size_t ndim,
    const ptrdiff_t *broadcasted_strides,
    const ptrdiff_t *target_strides) {

    size_t res = 0;
    for (size_t i = 0; i < ndim; ++i) {
        // Calculate contribution from each dimension
        res += flat_index / broadcasted_strides[i] * target_strides[i];
        // Remove the contribution from this dimension
        flat_index %= broadcasted_strides[i];
    }
    return res;
}

/**
 * @brief Converts a flattened index to a memory offset considering tensor striding.
 *
 * This is the general case for non-contiguous tensors where elements are not
 * stored sequentially in memory.
 *
 * @param flat_index The flattened index in the tensor
 * @param ndim Number of dimensions
 * @param shape Tensor shape
 * @param strides Tensor strides (in elements)
 * @return size_t Offset in the tensor's memory
 */
inline __mlu_device__ size_t indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {

    size_t res = 0;
    // Process dimensions from highest to lowest
    for (size_t i = ndim; i-- > 0;) {
        // Add contribution from this dimension
        res += (flat_index % shape[i]) * strides[i];
        // Remove the contribution from this dimension
        flat_index /= shape[i];
    }
    return res;
}

/**
 * @brief Helper struct for computing input tensor indices considering broadcasting and striding.
 *
 * This is particularly useful for operations where inputs may be broadcasted
 * to match the output shape, or may have non-contiguous memory layouts.
 */
struct InputIndexer {
    size_t idx;                      // Base index for this task
    size_t ndim;                     // Number of dimensions
    const bool *input_contiguous;    // Array indicating which inputs are contiguous
    const bool *input_broadcasted;   // Array indicating which inputs are broadcasted
    const size_t *input_shapes;      // Array of input shapes (concatenated)
    const ptrdiff_t *input_strides;  // Array of input strides (concatenated)
    const ptrdiff_t *output_strides; // Output tensor strides

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
                 ? global_idx // Simple case: contiguous memory
                 : (input_broadcasted[input_id]
                        // Handle broadcasted case
                        ? indexToReducedOffset(global_idx, ndim, output_strides, input_strides + input_id * ndim)
                        // General non-contiguous case
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
 * @brief Calculates optimal chunk size for memory operations based on tensor contiguity.
 *
 * This function identifies the largest contiguous block starting from the given
 * index and returns the maximum safe chunk size that can be processed in one
 * memory operation.
 *
 * @param global_idx_    Starting global index.
 * @param ndim           Number of dimensions.
 * @param shape          Tensor shape.
 * @param strides        Tensor strides.
 * @param max_len        Maximum allowed chunk size.
 * @return size_t        Optimal chunk size for memory operations.
 */
__mlu_device__ size_t calculateChunkSize(
    size_t global_idx_,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides,
    size_t max_len) {
    // Find the last dimension that is contiguous
    int last_contiguous_dim = -1;
    ptrdiff_t expected_stride = 1;

    // Check dimensions from highest to lowest
    for (int i = (int)ndim - 1; i >= 0; --i) {
        if (strides[i] != expected_stride) {
            break; // Stride doesn't match expected for contiguous layout
        }
        last_contiguous_dim = i;
        if (i > 0) {
            expected_stride *= shape[i];
        }
    }

    // If no contiguous dimension found, process one element at a time
    if (last_contiguous_dim < 0) {
        return 1;
    }

    // Calculate position in the contiguous block
    size_t global_idx = global_idx_;
    size_t pos_in_block = 0;
    size_t block_size = 1;

    for (int i = (int)ndim - 1; i >= last_contiguous_dim; --i) {
        size_t dim_idx = global_idx % shape[i];
        pos_in_block += dim_idx * block_size;
        block_size *= shape[i];
        global_idx /= shape[i];
    }

    // Calculate remaining elements in this contiguous block
    size_t remaining_in_block = block_size - pos_in_block;
    return std::min(max_len, remaining_in_block);
}

/**
 * @brief Checks if a memory segment is contiguous.
 *
 * @param start_idx     Starting index of the segment.
 * @param count         Number of elements in the segment.
 * @param ndim          Number of dimensions.
 * @param shape         Tensor shape.
 * @param strides       Tensor strides.
 * @return bool         True if the segment is contiguous in memory.
 */
__mlu_device__ bool isContiguous(
    size_t start_idx,
    size_t count,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {

    if (count <= 1) {
        return true;
    }

    // Verify the tensor follows contiguous memory layout
    ptrdiff_t expected_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape[i];
    }

    // Check if the segment is contiguous within this layout
    size_t end_idx = start_idx + count - 1;
    size_t linear_start = 0;
    size_t linear_end = 0;
    size_t temp_start = start_idx;
    size_t temp_end = end_idx;

    for (int i = 0; i < ndim; ++i) {
        size_t dim_size = shape[i];
        size_t start_coord = temp_start % dim_size;
        size_t end_coord = temp_end % dim_size;
        linear_start += start_coord * strides[i];
        linear_end += end_coord * strides[i];
        temp_start /= dim_size;
        temp_end /= dim_size;
    }

    // The segment is contiguous if the difference matches count-1 elements
    return (linear_end - linear_start) == (count - 1) * strides[ndim - 1];
}

/**
 * @brief Helper function for non-contiguous memory copy operations.
 *
 * This function handles copying data between NRAM (on-chip memory) and GDRAM
 * (global memory) for tensors with non-contiguous memory layouts. It uses
 * various strategies (single element copy, contiguous block copy, strided copy)
 * depending on the memory layout.
 *
 * @tparam Tdata        Data type of the elements.
 * @param dst           Destination buffer.
 * @param src           Source buffer.
 * @param direction     Memory copy direction (GDRAM2NRAM or NRAM2GDRAM).
 * @param indexer       Input indexer helper (for input copies).
 * @param input_idx     Input tensor index (for input copies).
 * @param processed     Number of elements already processed.
 * @param curr_batch    Current batch size.
 * @param start_idx     Starting index for this task.
 * @param ndim          Number of dimensions.
 * @param shape         Tensor shape.
 * @param strides       Tensor strides.
 * @param is_input_copy Whether this is an input copy operation.
 */
template <typename Tdata>
__mlu_device__ void nonContiguousMemcpy(
    Tdata *dst,
    Tdata *src,
    mluMemcpyDirection_t direction,
    InputIndexer &indexer,
    size_t input_idx,
    size_t processed,
    size_t curr_batch,
    size_t start_idx,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides,
    bool is_input_copy) {

    size_t remaining = curr_batch;
    size_t current_pos = 0;

    while (remaining > 0) {
        // Calculate current position in the tensor
        size_t global_idx = start_idx + processed + current_pos;

        // Get the element offset (either input or output)
        size_t element_offset = is_input_copy ? indexer(input_idx, processed + current_pos) : getOutputIndex(global_idx, false, ndim, shape, strides);

        // Calculate optimal chunk size for this segment
        size_t chunk_size = calculateChunkSize(
            global_idx, ndim, shape, strides, remaining);

        // Calculate strides in bytes
        ptrdiff_t src_stride_bytes = 0;
        ptrdiff_t dst_stride_bytes = 0;
        size_t segnum = 0;

        if (chunk_size > 1) {
            // For contiguous segments, use regular copy
            if (isContiguous(global_idx, chunk_size, ndim, shape, strides)) {
                __memcpy_async(
                    dst + (is_input_copy ? current_pos : element_offset),
                    src + (is_input_copy ? element_offset : current_pos),
                    chunk_size * sizeof(Tdata),
                    direction);
            }
            // For strided segments, use 2D memcpy
            else {
                // Calculate next element's offset to determine stride
                size_t next_offset = is_input_copy ? indexer(input_idx, processed + current_pos + 1) : getOutputIndex(global_idx + 1, false, ndim, shape, strides);

                if (is_input_copy) {
                    src_stride_bytes = (next_offset - element_offset) * sizeof(Tdata);
                    dst_stride_bytes = sizeof(Tdata); // NRAM is contiguous
                } else {
                    src_stride_bytes = sizeof(Tdata); // NRAM is contiguous
                    dst_stride_bytes = (next_offset - element_offset) * sizeof(Tdata);
                }

                // Number of segments is chunk_size - 1
                segnum = chunk_size - 1;

                __memcpy_async(
                    dst + (is_input_copy ? current_pos : element_offset),
                    src + (is_input_copy ? element_offset : current_pos),
                    sizeof(Tdata), // Size of each segment
                    direction,
                    dst_stride_bytes,
                    src_stride_bytes,
                    segnum);
            }
        } else {
            // Single element copy
            __memcpy_async(
                dst + (is_input_copy ? current_pos : element_offset),
                src + (is_input_copy ? element_offset : current_pos),
                sizeof(Tdata),
                direction);
        }

        current_pos += chunk_size;
        remaining -= chunk_size;
    }
}

} // namespace device::bang::kernel

#endif
