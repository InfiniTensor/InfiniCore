#include "infinicore/ops/concat_mla_q.hpp"
#include "../../utils.hpp"

#include <stdexcept>

#include "../vendor_ops/vendor_ops_dispatch.hpp"

namespace infinicore::op {

namespace {

void validate_concat_mla_q(const Tensor &ql_nope, const Tensor &q_pe, Tensor q_out) {
    if (!ql_nope || !q_pe || !q_out) {
        throw std::runtime_error("concat_mla_q expects non-empty ql_nope, q_pe, and q_out tensors");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(ql_nope, q_pe, q_out);
    if (ql_nope->dtype() != q_pe->dtype() || ql_nope->dtype() != q_out->dtype()) {
        throw std::runtime_error("concat_mla_q expects ql_nope, q_pe, and q_out to have the same dtype");
    }
    if (ql_nope->dtype() != DataType::F16 && ql_nope->dtype() != DataType::BF16 && ql_nope->dtype() != DataType::F32) {
        throw std::runtime_error("concat_mla_q expects float16, bfloat16, or float32 tensors");
    }
    if (ql_nope->ndim() != 3 || q_pe->ndim() != 3 || q_out->ndim() != 3) {
        throw std::runtime_error("concat_mla_q expects 3D tensors [tokens, heads, dim]");
    }
    if (ql_nope->size(0) != q_pe->size(0) || ql_nope->size(0) != q_out->size(0)
        || ql_nope->size(1) != q_pe->size(1) || ql_nope->size(1) != q_out->size(1)) {
        throw std::runtime_error("concat_mla_q expects matching first two dimensions");
    }
    if (ql_nope->size(2) + q_pe->size(2) != q_out->size(2)) {
        throw std::runtime_error("concat_mla_q expects q_out.shape[-1] == ql_nope.shape[-1] + q_pe.shape[-1]");
    }
    if (!ql_nope->is_contiguous() || !q_pe->is_contiguous() || !q_out->is_contiguous()) {
        throw std::runtime_error("concat_mla_q expects contiguous ql_nope, q_pe, and q_out tensors");
    }
}

} // namespace

void concat_mla_q_(const Tensor &ql_nope, const Tensor &q_pe, Tensor q_out) {
    validate_concat_mla_q(ql_nope, q_pe, q_out);

    auto kernel = vendor_ops::lookup(
        vendor_ops::concat_mla_q_dispatcher(),
        q_out->device().getType(),
        "concat_mla_q");
    kernel(ql_nope, q_pe, q_out);
}

Tensor concat_mla_q(const Tensor &ql_nope, const Tensor &q_pe) {
    if (!ql_nope || !q_pe) {
        throw std::runtime_error("concat_mla_q expects non-empty input tensors");
    }
    if (ql_nope->ndim() != 3 || q_pe->ndim() != 3) {
        throw std::runtime_error("concat_mla_q expects 3D tensors [tokens, heads, dim]");
    }
    Shape out_shape = ql_nope->shape();
    out_shape[2] = ql_nope->size(2) + q_pe->size(2);
    auto q_out = Tensor::empty(out_shape, ql_nope->dtype(), ql_nope->device());
    concat_mla_q_(ql_nope, q_pe, q_out);
    return q_out;
}

} // namespace infinicore::op
