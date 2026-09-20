#ifndef __BLOCK_FP8_GEMM_SM120_CUH__
#define __BLOCK_FP8_GEMM_SM120_CUH__

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cuda_runtime.h>
#include <cstddef>

namespace op::block_fp8_linear::nvidia::sm120 {

template <typename Kernel>
struct enable_sm120_family : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__)
  #if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
        Kernel::operator()(std::forward<Args>(args)...);
  #else
        printf("BlockFP8Linear: kernel only supports sm120 family.\n");
        asm("trap;");
  #endif
#endif
    }
};

struct GemmConfig {
    using ElementAB = cutlass::float_e4m3_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBlockScale = float;
    using ElementC = void;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementAB>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementAB>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
        1, 128, 128,
        cute::UMMA::Major::MN,
        cute::UMMA::Major::K>;

    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

    using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementBlockScale, RoundStyle>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute, ElementC,
        LayoutD, AlignmentD,
        ElementD, LayoutD, AlignmentD,
        EpilogueSchedule, DefaultOperation>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementAB,
        cute::tuple<LayoutA, LayoutSFA>,
        AlignmentA,
        ElementAB,
        cute::tuple<LayoutB, LayoutSFB>,
        AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>>;

    using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

inline size_t get_gemm_workspace_size(size_t M, size_t N, size_t K) {
    using GC = GemmConfig;
    using GemmOp = GC::GemmOp;
    using GemmKernel = typename GemmOp::GemmKernel;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;

    auto a_stride = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape((int)M, (int)K, 1));
    auto b_stride = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape((int)N, (int)K, 1));
    auto c_stride = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape((int)M, (int)N, 1));

    auto layout_SFA = GC::ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)M, (int)N, (int)K, 1));
    auto layout_SFB = GC::ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)M, (int)N, (int)K, 1));

    typename GemmKernel::MainloopArguments mainloop_args{};
    mainloop_args.ptr_A = nullptr;
    mainloop_args.dA = a_stride;
    mainloop_args.ptr_B = nullptr;
    mainloop_args.dB = b_stride;
    mainloop_args.ptr_SFA = nullptr;
    mainloop_args.ptr_SFB = nullptr;
    mainloop_args.layout_SFA = layout_SFA;
    mainloop_args.layout_SFB = layout_SFB;

    auto prob_shape = cute::make_shape((int)M, (int)N, (int)K, 1);

    typename GemmKernel::EpilogueArguments epilogue_args{
        {}, nullptr, c_stride, nullptr, c_stride};

    cutlass::KernelHardwareInfo hw_info;
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape, mainloop_args, epilogue_args, hw_info, {}};

    GemmOp gemm_op;
    size_t ws_size = gemm_op.get_workspace_size(args);
    return ws_size;
}

inline cutlass::Status run_gemm(
    void *output,
    const void *a_fp8,
    const void *a_scales,
    const void *weight,
    const void *weight_scales,
    size_t M, size_t N, size_t K,
    void *cutlass_workspace,
    cudaStream_t stream) {

    using GC = GemmConfig;
    using GemmOp = GC::GemmOp;
    using GemmKernel = typename GemmOp::GemmKernel;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;

    auto a_stride = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape((int)M, (int)K, 1));
    auto b_stride = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape((int)N, (int)K, 1));
    auto c_stride = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape((int)M, (int)N, 1));

    auto layout_SFA = GC::ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)M, (int)N, (int)K, 1));
    auto layout_SFB = GC::ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)M, (int)N, (int)K, 1));

    auto *a_ptr = static_cast<const typename GC::ElementAB *>(a_fp8);
    auto *b_ptr = static_cast<const typename GC::ElementAB *>(weight);
    auto *sfa_ptr = static_cast<const typename GC::ElementBlockScale *>(a_scales);
    auto *sfb_ptr = static_cast<const typename GC::ElementBlockScale *>(weight_scales);
    auto *d_ptr = static_cast<typename GC::ElementD *>(output);

    typename GemmKernel::MainloopArguments mainloop_args{};
    mainloop_args.ptr_A = a_ptr;
    mainloop_args.dA = a_stride;
    mainloop_args.ptr_B = b_ptr;
    mainloop_args.dB = b_stride;
    mainloop_args.ptr_SFA = sfa_ptr;
    mainloop_args.ptr_SFB = sfb_ptr;
    mainloop_args.layout_SFA = layout_SFA;
    mainloop_args.layout_SFB = layout_SFB;

    auto prob_shape = cute::make_shape((int)M, (int)N, (int)K, 1);

    typename GemmKernel::EpilogueArguments epilogue_args{
        {}, d_ptr, c_stride, d_ptr, c_stride};

    cutlass::KernelHardwareInfo hw_info;
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape, mainloop_args, epilogue_args, hw_info, {}};

    GemmOp gemm_op;

    auto can_impl = gemm_op.can_implement(args);
    if (can_impl != cutlass::Status::kSuccess) {
        return can_impl;
    }

    size_t ws_size = gemm_op.get_workspace_size(args);
    return gemm_op.run(args, cutlass_workspace, stream);
}

} // namespace op::block_fp8_linear::nvidia::sm120

#endif
