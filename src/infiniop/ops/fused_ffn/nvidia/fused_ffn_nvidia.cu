#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "fused_ffn_nvidia.cuh"
#include "kernel.cuh"

// Each op's public header re-defines a DESCRIPTOR(NAMESPACE) macro without
// guarding it, so including multiple sub-op headers in this TU clashes.
// We #undef between includes to sidestep the collision; order is arbitrary.
#undef DESCRIPTOR
#include "../../gemm/nvidia/gemm_nvidia.cuh"
#undef DESCRIPTOR
#include "../../rms_norm/nvidia/rms_norm_nvidia.cuh"

// swiglu / add use a separate ELEMENTWISE_DESCRIPTOR macro so they do not
// clash with the DESCRIPTOR macro above.
#include "../../add/nvidia/add_nvidia.cuh"
#include "../../swiglu/nvidia/swiglu_nvidia.cuh"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

namespace op::fused_ffn::nvidia {

namespace {

// 256-byte alignment is safe for both cublas working buffers and elementwise
// meta blobs; higher would waste workspace on small shapes.
constexpr size_t kWsAlign = 256;

inline size_t alignUp(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

// Heap-allocate a 2-D tensor descriptor with explicit element strides.
// Returned pointer must be deleted by the caller.
inline infiniopTensorDescriptor_t make2D(infiniDtype_t dtype,
                                         size_t d0, size_t d1,
                                         ptrdiff_t s0, ptrdiff_t s1) {
    const size_t shape[2] = {d0, d1};
    const ptrdiff_t strides[2] = {s0, s1};
    return new InfiniopTensorDescriptor(dtype, 2, shape, strides);
}

// Synthesize a GEMM B-matrix view with logical shape [k, n] regardless of
// whether the original weight was stored as [n, k] (Layout A) or [k, n]
// (Layout B). FusedFFNInfo::create already guarantees one of the two strides
// is 1, so this only needs to decide which dim is which and swap accordingly.
inline infiniopTensorDescriptor_t makeWeightAsKN(infiniDtype_t dtype,
                                                 size_t k, size_t n,
                                                 infiniopTensorDescriptor_t orig) {
    const size_t d0 = orig->dim(0);
    const ptrdiff_t s0 = orig->stride(0);
    const ptrdiff_t s1 = orig->stride(1);
    if (d0 == n) {
        // original [n, k] -> view as [k, n] by swapping axes
        return make2D(dtype, k, n, s1, s0);
    }
    // original already [k, n]
    return make2D(dtype, k, n, s0, s1);
}

// RAII wrapper: owns a list of synthesized tensor descriptors and deletes
// them on scope exit. Sub-descriptor create() calls copy out what they need,
// so the temporaries only need to outlive the create() call.
class DescScope {
    std::vector<infiniopTensorDescriptor_t> _owned;

public:
    ~DescScope() {
        for (auto *t : _owned) {
            delete t;
        }
    }
    infiniopTensorDescriptor_t adopt(infiniopTensorDescriptor_t t) {
        _owned.push_back(t);
        return t;
    }
};

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;

    // Workspace slab sizes (bytes), padded to kWsAlign.
    size_t normalized_bytes = 0;
    size_t gate_up_bytes = 0;
    size_t hidden_bytes = 0;
    size_t inner_ws_bytes = 0; // max of sub-descriptor workspaceSize()

    bool has_residual = false;

    // Deep-fused Gate-Up + SwiGLU (paper-style fusion) config.
    // Decided at create-time via env var INFINIOP_FUSED_FFN_DEEP and
    // a profile-driven scheduler that checks ntok against a threshold:
    //   INFINIOP_FUSED_FFN_DEEP=0 or unset -> always 5-stage (default)
    //   INFINIOP_FUSED_FFN_DEEP=1          -> scheduler: deep-fused when
    //       ntok <= threshold, else 5-stage. Threshold defaults to 4 and
    //       can be overridden with INFINIOP_FUSED_FFN_DEEP_MAX_NTOK.
    //   INFINIOP_FUSED_FFN_DEEP=2          -> force deep-fused always
    //       (debug/profiling only; will regress at large ntok)
    bool use_deep_fused = false;
    ptrdiff_t gate_up_w_k_stride = 0;
    ptrdiff_t gate_up_w_col_stride = 0;

    // Sub-descriptors owned by this fused op; each one is a standard
    // InfiniopDescriptor for the corresponding standalone operator.
    std::unique_ptr<op::rms_norm::nvidia::Descriptor> rms_norm;
    std::unique_ptr<op::gemm::nvidia::Descriptor> gate_up_gemm;
    std::unique_ptr<op::swiglu::nvidia::Descriptor> swiglu;
    std::unique_ptr<op::gemm::nvidia::Descriptor> down_gemm;
    std::unique_ptr<op::add::nvidia::Descriptor> residual_add;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    infiniopTensorDescriptor_t residual_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t gate_up_weight_desc,
    infiniopTensorDescriptor_t down_weight_desc,
    float epsilon) {

    auto info_result = FusedFFNInfo::create(
        out_desc, in_desc, residual_desc,
        norm_weight_desc, gate_up_weight_desc, down_weight_desc, epsilon);
    CHECK_RESULT(info_result);
    auto info = info_result.take();

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto opaque = std::make_unique<Opaque>();
    opaque->internal = handle->internal();
    opaque->has_residual = info.has_residual;

    const size_t ntok = info.ntok();
    const size_t d = info.d();
    const size_t di = info.di();
    const size_t dtype_sz = infiniSizeOf(info.dtype);

    // Profile-driven scheduler for the deep-fused kernel path.
    //   INFINIOP_FUSED_FFN_DEEP=0/unset -> always 5-stage (default)
    //   INFINIOP_FUSED_FFN_DEEP=1       -> scheduler: use deep-fused only
    //       when ntok <= max_ntok threshold (default 4); fall back to
    //       5-stage for larger shapes where cuBLAS tensor-core GEMM wins.
    //   INFINIOP_FUSED_FFN_DEEP=2       -> force deep-fused always (debug)
    // Threshold is tunable via INFINIOP_FUSED_FFN_DEEP_MAX_NTOK (default 4).
    {
        const char *env = std::getenv("INFINIOP_FUSED_FFN_DEEP");
        if (env != nullptr && env[0] == '2') {
            // Mode 2: force deep-fused regardless of shape
            opaque->use_deep_fused = true;
        } else if (env != nullptr && env[0] == '1') {
            // Mode 1: scheduler — deep-fused only for small ntok
            size_t max_ntok = 4;
            const char *thr = std::getenv("INFINIOP_FUSED_FFN_DEEP_MAX_NTOK");
            if (thr != nullptr) {
                max_ntok = static_cast<size_t>(std::atol(thr));
                if (max_ntok == 0) {
                    max_ntok = 4;
                }
            }
            opaque->use_deep_fused = (ntok <= max_ntok);
        }
        // Mode 0 / unset: use_deep_fused stays false (5-stage)
    }

    // Extract gate_up weight strides at create time so the deep-fused kernel
    // can index [k, j] regardless of whether storage is Layout A [2*di, d]
    // or Layout B [d, 2*di]. Layout is identified by which descriptor dim
    // equals 2*di (the output column axis) vs d (the K axis).
    {
        const size_t gu_dim0 = gate_up_weight_desc->dim(0);
        const ptrdiff_t gu_s0 = gate_up_weight_desc->stride(0);
        const ptrdiff_t gu_s1 = gate_up_weight_desc->stride(1);
        if (gu_dim0 == 2 * di) {
            // Layout A: [2*di, d] — output column is dim0, K is dim1.
            opaque->gate_up_w_col_stride = gu_s0;
            opaque->gate_up_w_k_stride = gu_s1;
        } else {
            // Layout B: [d, 2*di] — K is dim0, output column is dim1.
            opaque->gate_up_w_k_stride = gu_s0;
            opaque->gate_up_w_col_stride = gu_s1;
        }
    }

    // ── Workspace layout ──
    //   normalized : [ntok, d]     contiguous   -> RMSNorm out,   GateUp in
    //   gate_up    : [ntok, 2*di]  contiguous   -> GateUp out,    SwiGLU in
    //   hidden     : [ntok, di]    contiguous   -> SwiGLU out,    Down  in
    //   inner_ws   : max(sub->workspaceSize())  shared by sub-descriptors
    //
    // The compact hidden slab (stride=di instead of stride=2*di) gives the
    // Down-GEMM a tightly packed K dimension, which matters on BIV150 where
    // cuBLAS 10.2 tensor-core paths prefer aligned contiguous leading dims.
    opaque->normalized_bytes = alignUp(ntok * d * dtype_sz, kWsAlign);
    opaque->gate_up_bytes = alignUp(ntok * 2 * di * dtype_sz, kWsAlign);
    opaque->hidden_bytes = alignUp(ntok * di * dtype_sz, kWsAlign);

    DescScope scope;

    // ── RMSNorm sub-descriptor ──
    // Activation is 2-D [ntok, d]; weight is 1-D [d].
    auto normalized_desc = scope.adopt(
        make2D(info.dtype, ntok, d, static_cast<ptrdiff_t>(d), 1));
    auto in_view = scope.adopt(
        make2D(info.dtype, ntok, d, info.in_stride, 1));

    {
        op::rms_norm::nvidia::Descriptor *sub = nullptr;
        CHECK_STATUS(op::rms_norm::nvidia::Descriptor::create(
            handle_, &sub,
            normalized_desc, in_view, norm_weight_desc,
            info.epsilon));
        opaque->rms_norm.reset(sub);
        opaque->inner_ws_bytes = std::max(opaque->inner_ws_bytes, sub->workspaceSize());
    }

    // ── GateUp GEMM sub-descriptor ──
    //   [ntok, 2*di] = [ntok, d] @ [d, 2*di]
    auto gate_up_c_desc = scope.adopt(
        make2D(info.dtype, ntok, 2 * di, static_cast<ptrdiff_t>(2 * di), 1));
    auto gate_up_b_desc = scope.adopt(
        makeWeightAsKN(info.mtype, d, 2 * di, gate_up_weight_desc));

    {
        op::gemm::nvidia::Descriptor *sub = nullptr;
        CHECK_STATUS(op::gemm::nvidia::Descriptor::create(
            handle_, &sub, gate_up_c_desc, normalized_desc, gate_up_b_desc));
        opaque->gate_up_gemm.reset(sub);
        opaque->inner_ws_bytes = std::max(opaque->inner_ws_bytes, sub->workspaceSize());
    }

    // ── SwiGLU sub-descriptor ──
    // Operates on the interleaved [gate | up] buffer:
    //   logical inputs : up   [ntok, di] row stride 2*di
    //                    gate [ntok, di] row stride 2*di
    //   logical output : hidden [ntok, di] contiguous
    // gate and up share identical shape/strides — only their base pointers
    // differ at calculate time.
    auto hidden_desc = scope.adopt(
        make2D(info.dtype, ntok, di, static_cast<ptrdiff_t>(di), 1));
    auto half_desc = scope.adopt(
        make2D(info.dtype, ntok, di, static_cast<ptrdiff_t>(2 * di), 1));

    {
        op::swiglu::nvidia::Descriptor *sub = nullptr;
        CHECK_STATUS(op::swiglu::nvidia::Descriptor::create(
            handle_, &sub, hidden_desc, {half_desc, half_desc}));
        opaque->swiglu.reset(sub);
        opaque->inner_ws_bytes = std::max(opaque->inner_ws_bytes, sub->workspaceSize());
    }

    // ── Down GEMM sub-descriptor ──
    //   out = [beta * out] + 1.0 * hidden @ down_weight
    // The output matrix uses the user's out stride so the gemm writes
    // directly into the caller's tensor.
    auto out_view = scope.adopt(
        make2D(info.dtype, ntok, d, info.out_stride, 1));
    auto down_b_desc = scope.adopt(
        makeWeightAsKN(info.mtype, di, d, down_weight_desc));

    {
        op::gemm::nvidia::Descriptor *sub = nullptr;
        CHECK_STATUS(op::gemm::nvidia::Descriptor::create(
            handle_, &sub, out_view, hidden_desc, down_b_desc));
        opaque->down_gemm.reset(sub);
        opaque->inner_ws_bytes = std::max(opaque->inner_ws_bytes, sub->workspaceSize());
    }

    // ── Residual add sub-descriptor (optional) ──
    // Only used when residual is a distinct tensor from out; the
    // (out == residual) case is fused into the Down-GEMM via beta=1 at
    // calculate time.
    if (info.has_residual) {
        auto residual_view = scope.adopt(
            make2D(info.dtype, ntok, d, info.residual_stride, 1));
        auto out_view_for_add = scope.adopt(
            make2D(info.dtype, ntok, d, info.out_stride, 1));

        op::add::nvidia::Descriptor *sub = nullptr;
        CHECK_STATUS(op::add::nvidia::Descriptor::create(
            handle_, &sub,
            out_view_for_add, {out_view_for_add, residual_view}));
        opaque->residual_add.reset(sub);
        opaque->inner_ws_bytes = std::max(opaque->inner_ws_bytes, sub->workspaceSize());
    }

    const size_t workspace_size = opaque->normalized_bytes + opaque->gate_up_bytes + opaque->hidden_bytes + alignUp(opaque->inner_ws_bytes, kWsAlign);

    *desc_ptr = new Descriptor(
        opaque.release(),
        std::move(info),
        workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *in,
    const void *residual,
    const void *norm_weight,
    const void *gate_up_weight,
    const void *down_weight,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    const size_t di = _info.di();
    const size_t dtype_sz = infiniSizeOf(_info.dtype);

    // Partition the workspace into the three persistent slabs plus an
    // inner scratch buffer shared by all sub-descriptors.
    char *ws = static_cast<char *>(workspace);
    void *normalized_buf = ws;
    ws += _opaque->normalized_bytes;
    void *gate_up_buf = ws;
    ws += _opaque->gate_up_bytes;
    void *hidden_buf = ws;
    ws += _opaque->hidden_bytes;
    void *inner_ws = ws;
    const size_t inner_ws_size = _opaque->inner_ws_bytes;

    // gate and up are two halves of the interleaved gate_up buffer.
    // Each token's layout is [gate[0..di) | up[0..di)], so gate starts at
    // offset 0 and up starts at offset (di * dtype_sz) bytes within the
    // first row; both use a row stride of 2*di elements (captured in the
    // shared half_desc at create time).
    const char *gu_bytes = static_cast<const char *>(gate_up_buf);
    const void *gate_ptr = gu_bytes;
    const void *up_ptr = gu_bytes + di * dtype_sz;

    // Stage 1: RMSNorm
    CHECK_STATUS(_opaque->rms_norm->calculate(
        inner_ws, inner_ws_size,
        normalized_buf, in, norm_weight, stream));

    if (_opaque->use_deep_fused) {
        // Stage 2+3 fused: one kernel produces hidden = SiLU(norm@Wg) * (norm@Wu)
        // directly, eliminating the gate_up_buf HBM round-trip. Row strides
        // for X and hidden are d and di respectively (contiguous buffers
        // allocated at create time).
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        const size_t ntok = _info.ntok();
        const size_t d = _info.d();
        const size_t di = _info.di();

        constexpr unsigned int kBlock = 256;
        dim3 grid(static_cast<unsigned>(ntok), static_cast<unsigned>(di));
        dim3 block(kBlock);

#define DEEP_FUSED_LAUNCH(TD, TW)                         \
    deepFusedGateUpSiluKernel<kBlock, float, TD, TW>      \
        <<<grid, block, 0, cuda_stream>>>(                \
            reinterpret_cast<TD *>(hidden_buf),           \
            reinterpret_cast<const TD *>(normalized_buf), \
            reinterpret_cast<const TW *>(gate_up_weight), \
            ntok, d, di,                                  \
            static_cast<ptrdiff_t>(d),                    \
            static_cast<ptrdiff_t>(di),                   \
            _opaque->gate_up_w_k_stride,                  \
            _opaque->gate_up_w_col_stride,                \
            /*gate_col_base=*/0u, /*up_col_base=*/di)

        if (_info.dtype == INFINI_DTYPE_F16 && _info.mtype == INFINI_DTYPE_F16) {
            DEEP_FUSED_LAUNCH(half, half);
        } else if (_info.dtype == INFINI_DTYPE_BF16 && _info.mtype == INFINI_DTYPE_BF16) {
            DEEP_FUSED_LAUNCH(__nv_bfloat16, __nv_bfloat16);
        } else if (_info.dtype == INFINI_DTYPE_F32 && _info.mtype == INFINI_DTYPE_F32) {
            DEEP_FUSED_LAUNCH(float, float);
        } else if (_info.dtype == INFINI_DTYPE_F16 && _info.mtype == INFINI_DTYPE_F32) {
            DEEP_FUSED_LAUNCH(half, float);
        } else if (_info.dtype == INFINI_DTYPE_BF16 && _info.mtype == INFINI_DTYPE_F32) {
            DEEP_FUSED_LAUNCH(__nv_bfloat16, float);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
#undef DEEP_FUSED_LAUNCH
        // Avoid unused-variable warnings for the baseline-only pointers when
        // the deep-fused branch takes over stage 2/3.
        (void)gate_up_buf;
        (void)gate_ptr;
        (void)up_ptr;
    } else {
        // Stage 2: GateUp GEMM  -->  gate_up_buf = normalized_buf @ gate_up_weight
        CHECK_STATUS(_opaque->gate_up_gemm->calculate(
            inner_ws, inner_ws_size,
            gate_up_buf, /*beta=*/0.f,
            normalized_buf, gate_up_weight,
            /*alpha=*/1.f, stream));

        // Stage 3: SwiGLU  -->  hidden_buf = silu(gate) * up
        // swiglu::nvidia expects inputs ordered {up, gate}; see
        // swiglu_nvidia.cu input_desc_vec[0]=up, [1]=gate.
        CHECK_STATUS(_opaque->swiglu->calculate(
            inner_ws, inner_ws_size,
            hidden_buf, {up_ptr, gate_ptr}, stream));
    }

    // Stage 4: Down GEMM, with optional in-place residual fuse via beta=1.
    //   fuse path : out = 1.0 * out + hidden_buf @ down_weight
    //   plain path: out = 0.0 * out + hidden_buf @ down_weight
    const bool fuse_residual = _opaque->has_residual && (out == residual);
    CHECK_STATUS(_opaque->down_gemm->calculate(
        inner_ws, inner_ws_size,
        out, /*beta=*/fuse_residual ? 1.f : 0.f,
        hidden_buf, down_weight,
        /*alpha=*/1.f, stream));

    // Stage 5: Residual add (only when the in-place fuse did not apply).
    if (_opaque->has_residual && !fuse_residual) {
        CHECK_STATUS(_opaque->residual_add->calculate(
            inner_ws, inner_ws_size,
            out, {out, residual}, stream));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::fused_ffn::nvidia
