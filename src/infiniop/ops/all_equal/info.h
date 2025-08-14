#ifndef __ALL_EQUAL_INFO_H__
#define __ALL_EQUAL_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infinicore.h"
#include <cstddef>
#include <cstring>
#include <vector>

namespace op::all_equal {
struct AllEqualInfo {
private:
    std::vector<size_t> _meta;
    size_t _input_size;
    size_t _input_numel;
    size_t _ndim;

    AllEqualInfo(std::vector<size_t> meta,
                 size_t input_size,
                 size_t input_numel,
                 size_t ndim,
                 infiniDtype_t dtype)
        : _meta(std::move(meta)),
          _input_size(input_size), _input_numel(input_numel), _ndim(ndim), _dtype(dtype) {}

public:
    infiniDtype_t _dtype;

public:
    // Get the Memory size of the meta data in bytes
    inline size_t getMetaMemSize() const {
        return _meta.size() * sizeof(size_t);
    }
    inline const int8_t *getMetaStart() const {
        return reinterpret_cast<const int8_t *>(_meta.data());
    }
    inline size_t getInputSize() const {
        return _input_size;
    }
    inline size_t getInputNumel() const {
        return _input_numel;
    }
    inline size_t getNdim() const {
        return _ndim;
    }
    inline const ptrdiff_t *getDefaultStrides() const {
        return reinterpret_cast<const ptrdiff_t *>(_meta.data());
    }
    inline const size_t *getAllInputShapes() const {
        return reinterpret_cast<const size_t *>(getDefaultStrides() + _ndim);
    }
    inline const size_t *getInputShape(const size_t &index) const {
        if (index < _input_size) {
            return reinterpret_cast<const size_t *>(getAllInputShapes() + index * _ndim);
        }
        return nullptr;
    }
    inline const ptrdiff_t *getAllInputStrides() const {
        return reinterpret_cast<const ptrdiff_t *>(getAllInputShapes() + _input_size * _ndim);
    }
    inline const ptrdiff_t *getInputStrides(const size_t &index) const {
        if (index < _input_size) {
            return reinterpret_cast<const ptrdiff_t *>(getAllInputStrides() + index * _ndim);
        }
        return nullptr;
    }
    inline const bool *getInputContiguous() const {
        return reinterpret_cast<const bool *>(getAllInputStrides() + _input_size * _ndim);
    }
    inline const bool *getInputBroadcasted() const {
        return reinterpret_cast<const bool *>(getInputContiguous() + _input_size);
    }

    using ResultType = utils::Result<AllEqualInfo>;

    /**
     * @brief Construct ElementwiseInfo from output and input tensor descriptors.
     * @param output_desc Descriptor of the output tensor.
     * @param input_descs Descriptors of the input tensors.
     * @return Result<ElementwiseInfo> with the successfully constructed ElementwiseInfo,
     *         or the status code.
     */
    static ResultType create(
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs) {

        if (!output_desc || input_descs.empty()) {
            return INFINI_STATUS_BAD_PARAM;
        }

        // Destination cannot have broadcast setup
        if (output_desc->hasBroadcastDim()) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        auto input_size = input_descs.size();
        auto input_a_desc = input_descs[0];
        auto input_numel = input_a_desc->numel();
        auto ndim = input_a_desc->ndim();
        auto dtype = input_a_desc->dtype();

        // Allocate memory for meta
        auto shape_unit = input_a_desc->dim(0);
        auto stride_unit = input_a_desc->stride(0);
        size_t meta_mem_size = ndim * sizeof(stride_unit) + input_size * ndim * sizeof(shape_unit)
                             + input_size * ndim * sizeof(stride_unit)
                             + 2 * input_size * sizeof(bool);
        std::vector<size_t> meta(CEIL_DIV(meta_mem_size, sizeof(size_t)));
        int8_t *meta_ptr = reinterpret_cast<int8_t *>(meta.data());

        std::vector<ptrdiff_t> default_strides(ndim);
        auto default_shape = input_a_desc->shape();
        ptrdiff_t dsize = 1;
        for (int i = (int)ndim - 1; i >= 0; i--) {
            default_strides[i] = dsize;
            dsize *= default_shape[i];
        }

        // Pointers to the sections within _meta
        size_t *default_strides_p = reinterpret_cast<size_t *>(meta_ptr);
        size_t *input_shapes = reinterpret_cast<size_t *>(default_strides_p + ndim);
        ptrdiff_t *input_strides = reinterpret_cast<ptrdiff_t *>(input_shapes + input_size * ndim);
        bool *input_contiguous = reinterpret_cast<bool *>(input_strides + input_size * ndim);
        bool *input_broadcasted = input_contiguous + input_size;

        // Copy default strides
        std::memcpy(default_strides_p, default_strides.data(), ndim * sizeof(*default_strides_p));

        // Copy input shapes, strides, contiguous, and broadcasted flags
        for (size_t i = 0; i < input_size; ++i) {
            auto &desc = input_descs[i];
            const auto in_shape = desc->shape();
            const auto in_strides = desc->strides();
            std::memcpy(input_shapes + i * ndim, in_shape.data(), ndim * sizeof(*input_shapes));
            std::memcpy(input_strides + i * ndim, in_strides.data(), ndim * sizeof(*input_strides));
            input_contiguous[i] = desc->isContiguous();
            input_broadcasted[i] = !input_contiguous[i] && desc->hasBroadcastDim();
        }

        AllEqualInfo info(std::move(meta), input_size, input_numel, ndim, dtype);
        return ResultType(std::move(info));
    }
};

} // namespace op::all_equal

#endif // __ALL_EQUAL_INFO_H__
