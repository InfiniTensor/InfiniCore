#include "infinicore/ops/vision_qkv_rope_pad.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(VisionQKVRoPEPad);
VisionQKVRoPEPad::VisionQKVRoPEPad(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &qkv, const Tensor &cos, const Tensor &sin, const Tensor &rows, const Tensor &cols) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_out, k_out, v_out, qkv, cos, sin, rows, cols);
    INFINICORE_GRAPH_OP_DISPATCH(qkv->device().getType(), q_out, k_out, v_out, qkv, cos, sin, rows, cols);
}
void VisionQKVRoPEPad::execute(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &qkv, const Tensor &cos, const Tensor &sin, const Tensor &rows, const Tensor &cols) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(VisionQKVRoPEPad, q_out, k_out, v_out, qkv, cos, sin, rows, cols);
}
void vision_qkv_rope_pad_(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &qkv, const Tensor &cos, const Tensor &sin, const Tensor &rows, const Tensor &cols) { VisionQKVRoPEPad::execute(q_out, k_out, v_out, qkv, cos, sin, rows, cols); }
} // namespace infinicore::op
