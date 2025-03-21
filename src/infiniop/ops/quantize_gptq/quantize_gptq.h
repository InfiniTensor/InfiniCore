#ifndef __QUANTIZE_GPTQ_H__
#define __QUANTIZE_GPTQ_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#define DESCRIPTOR(NAMESPACE)                                                  \
                                                                               \
    namespace op::quantize_gptq::NAMESPACE {                                   \
    class Descriptor final : public InfiniopDescriptor {                       \
        struct Opaque;                                                         \
        Opaque *_opaque;                                                       \
        MatmulGptqInfo _info;                                                  \
        size_t _workspace_size;                                                \
                                                                               \
        Descriptor(MatmulGptqInfo info, Opaque *opaque,                        \
                   size_t workspace_size,                                      \
                   infiniDevice_t device_type, int device_id)                  \
            : InfiniopDescriptor{device_type, device_id},                      \
              _opaque(opaque), _info(info), _workspace_size(workspace_size) {} \
                                                                               \
    public:                                                                    \
        ~Descriptor();                                                         \
                                                                               \
        size_t minWorkspaceSize() const { return _workspace_size; }            \
                                                                               \
        static infiniStatus_t create(                                          \
            infiniopHandle_t handle, Descriptor **desc_ptr,                    \
            infiniopTensorDescriptor_t c_desc,                                 \
            infiniopTensorDescriptor_t a_desc,                                 \
            infiniopTensorDescriptor_t packed_weights_desc,                    \
            infiniopTensorDescriptor_t b_scale_desc,                           \
            infiniopTensorDescriptor_t zero_desc);                             \
                                                                               \
        infiniStatus_t quant(                                                  \
            void *workspace, size_t workspace_size,                            \
            void *packed_weights, void *b_scale, void *zero,                   \
            const void *a, const void *b, void *stream) const;                 \
                                                                               \
        infiniStatus_t calculate(                                              \
            void *workspace, size_t workspace_size,                            \
            void *c, const void *a,                                            \
            void *packed_weights, void *b_scale,                               \
            void *zero, void *stream) const;                                   \
    };                                                                         \
    }

class MatmulGptqInfo {
private:
    MatmulGptqInfo() = default;

public:
    infiniDtype_t atype, packed_weights_type;
    size_t m, k, n, num_groups, block_size;
    ptrdiff_t group_size;
    bool is_weight_transposed;

    static utils::Result<MatmulGptqInfo> createMatmulGptqInfo(
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t packed_weights_desc,
        infiniopTensorDescriptor_t b_scale_desc,
        infiniopTensorDescriptor_t zero_desc) {

        CHECK_OR_RETURN(
            c_desc != nullptr && a_desc != nullptr && packed_weights_desc != nullptr && b_scale_desc != nullptr && zero_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t atype = a_desc->dtype();
        const infiniDtype_t packed_weights_type = packed_weights_desc->dtype();
        CHECK_OR_RETURN(atype == c_desc->dtype() && atype == b_scale_desc->dtype() && atype == zero_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(atype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
        CHECK_DTYPE(packed_weights_type, INFINI_DTYPE_I32);

        CHECK_OR_RETURN(c_desc->ndim() == 2
                            && a_desc->ndim() == 2
                            && packed_weights_desc->ndim() == 2
                            && b_scale_desc->ndim() == 2
                            && zero_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        bool is_weight_transposed = false;
        size_t m = 1;
        size_t k = 1;
        size_t n = 1;
        size_t num_groups = 1;
        CHECK_OR_RETURN(c_desc->dim(0) == a_desc->dim(0)
                            || c_desc->dim(1) == a_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        if (c_desc->dim(0) == a_desc->dim(0)) {
            if (c_desc->dim(1) == a_desc->dim(1)) {
                if (packed_weights_desc->dim(0) * 8 == packed_weights_desc->dim(1)) {
                    is_weight_transposed = true;
                    m = c_desc->dim(0);
                    n = c_desc->dim(1);
                    k = a_desc->dim(1);
                    num_groups = b_scale_desc->dim(0);
                } else if (packed_weights_desc->dim(0) == packed_weights_desc->dim(1) * 8) {
                    is_weight_transposed = false;
                    m = c_desc->dim(1);
                    n = c_desc->dim(0);
                    k = a_desc->dim(0);
                    num_groups = b_scale_desc->dim(1);
                }
            } else {
                is_weight_transposed = true;
                m = c_desc->dim(0);
                n = c_desc->dim(1);
                k = a_desc->dim(1);
                num_groups = b_scale_desc->dim(0);
            }

        } else { // c_desc->dim(0) != a_desc->dim(0)
            if (c_desc->dim(1) == a_desc->dim(1)) {
                is_weight_transposed = false;
                m = c_desc->dim(1);
                n = c_desc->dim(0);
                k = a_desc->dim(0);
                num_groups = b_scale_desc->dim(1);
            }
        }

        size_t block_size = 128;
        ptrdiff_t group_size = num_groups > 1 ? static_cast<ptrdiff_t>(k) / static_cast<ptrdiff_t>(num_groups) : -1;
        const size_t k_8 = k / 8;
        if (is_weight_transposed) {
            CHECK_OR_RETURN(m == a_desc->dim(0)
                                && num_groups == zero_desc->dim(0)
                                && n == b_scale_desc->dim(1) && n == zero_desc->dim(1)
                                && n == packed_weights_desc->dim(1) && k_8 == packed_weights_desc->dim(0),
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        } else {
            CHECK_OR_RETURN(m == a_desc->dim(1)
                                && num_groups == zero_desc->dim(1)
                                && n == b_scale_desc->dim(0) && n == zero_desc->dim(0)
                                && n == packed_weights_desc->dim(0) && k_8 == packed_weights_desc->dim(1),
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        return utils::Result<MatmulGptqInfo>(MatmulGptqInfo{
            atype,
            packed_weights_type,
            m,
            k,
            n,
            num_groups,
            block_size,
            group_size,
            is_weight_transposed,
        });
    }
};

#endif // __QUANTIZE_GPTQ_H__
