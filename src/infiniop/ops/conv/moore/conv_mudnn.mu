#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_handle.h"
#include "conv_mudnn.h"

#include <musa_bf16.h>

namespace op::conv::mudnn {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    const void *pads,
    const void *strides,
    const void *dilations,
    size_t n) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc, b_desc, pads, strides, dilations, n);
    CHECK_RESULT(result);

    auto info = result.take();

    *desc_ptr = new Descriptor(
        dtype, info, 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculate(
    const ConvInfo &info,
    std::shared_ptr<device::moore::Handle::Internal> &_internal,
    void *y,
    const void *x,
    const void *w,
    const void *bias,
    void *stream) {

    auto conv_operator = std::make_unique<::musa::dnn::Convolution>();
    conv_operator->SetComputeMode(::musa::dnn::Convolution::ComputeMode::TENSOR);

    // Use muDNN handle management
    return _internal->useMudnn((musaStream_t)stream, [&](::musa::dnn::Handle &mudnn_handle) -> infiniStatus_t {

        // 3. Create Tensor
        ::musa::dnn::Tensor input_tensor, output_tensor, weight_tensor, bias_tensor;

        if constexpr (std::is_same<Tdata, half>::value) {
            input_tensor.SetType(::musa::dnn::Tensor::Type::HALF);
            output_tensor.SetType(::musa::dnn::Tensor::Type::HALF);
            weight_tensor.SetType(::musa::dnn::Tensor::Type::HALF);
            bias_tensor.SetType(::musa::dnn::Tensor::Type::HALF);
        } else if constexpr (std::is_same<Tdata, __mt_bfloat16>::value) {
            input_tensor.SetType(::musa::dnn::Tensor::Type::BFLOAT16);
            output_tensor.SetType(::musa::dnn::Tensor::Type::BFLOAT16);
            weight_tensor.SetType(::musa::dnn::Tensor::Type::BFLOAT16);
            bias_tensor.SetType(::musa::dnn::Tensor::Type::BFLOAT16);
        } else {
            input_tensor.SetType(::musa::dnn::Tensor::Type::FLOAT);
            output_tensor.SetType(::musa::dnn::Tensor::Type::FLOAT);
            weight_tensor.SetType(::musa::dnn::Tensor::Type::FLOAT);
            bias_tensor.SetType(::musa::dnn::Tensor::Type::FLOAT);
        }

        // 4. Bind Tensor addr
        input_tensor.SetAddr(const_cast<void*>(x));
        output_tensor.SetAddr(y);
        weight_tensor.SetAddr(const_cast<void*>(w));
        bias_tensor.SetAddr(const_cast<void*>(bias));
{
        // 5. Config Tensor input_tensor: [N, C, spatial...]
        const size_t ndim = info.ndim();
        std::vector<int64_t> x_dims;
        x_dims.reserve(ndim + 2);

        x_dims.push_back(static_cast<int64_t>(info.batch()));
        x_dims.push_back(static_cast<int64_t>(info.in_channels()));
        for (size_t i = 0; i < ndim; ++i) {
            x_dims.push_back(static_cast<int64_t>(info.input_dim(i)));
        }

        // contiguous stride
        std::vector<int64_t> x_stride(x_dims.size());
        x_stride.back() = 1;
        for (int i = static_cast<int>(x_dims.size()) - 2; i >= 0; --i) {
            x_stride[i] = x_stride[i + 1] * x_dims[i + 1];
        }

        input_tensor.SetNdInfo(
            static_cast<int>(x_dims.size()),
            x_dims.data(),
            x_stride.data()
        );

}
{
        // 6. Config Tensor weight_tensor: [Cout, Cin, kernel...]
        const size_t ndim = info.ndim();
        std::vector<int64_t> w_dims;
        w_dims.reserve(ndim + 2);

        w_dims.push_back(static_cast<int64_t>(info.out_channels()));
        w_dims.push_back(static_cast<int64_t>(info.in_channels())); // groups=1
        for (size_t i = 0; i < ndim; ++i) {
            w_dims.push_back(static_cast<int64_t>(info.kernel_dim(i)));
        }

        std::vector<int64_t> w_stride(w_dims.size());
        w_stride.back() = 1;
        for (int i = static_cast<int>(w_dims.size()) - 2; i >= 0; --i) {
            w_stride[i] = w_stride[i + 1] * w_dims[i + 1];
        }

        weight_tensor.SetNdInfo(
            static_cast<int>(w_dims.size()),
            w_dims.data(),
            w_stride.data()
        );

}
{
        // 7. Config Tensor output_tensor: [N, Cout, spatial...]
        const size_t ndim = info.ndim();
        std::vector<int64_t> y_dims;
        y_dims.reserve(ndim + 2);

        y_dims.push_back(static_cast<int64_t>(info.batch()));
        y_dims.push_back(static_cast<int64_t>(info.out_channels()));
        for (size_t i = 0; i < ndim; ++i) {
            y_dims.push_back(static_cast<int64_t>(info.output_dim(i)));
        }

        std::vector<int64_t> y_stride(y_dims.size());
        y_stride.back() = 1;
        for (int i = static_cast<int>(y_dims.size()) - 2; i >= 0; --i) {
            y_stride[i] = y_stride[i + 1] * y_dims[i + 1];
        }

        output_tensor.SetNdInfo(
            static_cast<int>(y_dims.size()),
            y_dims.data(),
            y_stride.data()
        );
}

        // 8. Bias tensor (if exists)
        if (bias != nullptr) {
            std::array<int64_t, 1> b_dims = {
                static_cast<int64_t>(info.out_channels())
            };
            std::array<int64_t, 1> b_stride = {1};
            bias_tensor.SetNdInfo(1, b_dims.data(), b_stride.data());
        }

        // 9. Configure convolution descriptor (from ConvInfo)
        std::vector<int> pad_dims(info.ndim());
        std::vector<int> stride_dims(info.ndim());
        std::vector<int> dilation_dims(info.ndim());

        for (size_t i = 0; i < info.ndim(); ++i) {
            pad_dims[i]       = static_cast<int>(info.pad_info(i));
            stride_dims[i]    = static_cast<int>(info.stride_info(i));
            dilation_dims[i]  = static_cast<int>(info.dilation_info(i));
        }

        // Current infiniop ConvInfo implies groups == 1
        conv_operator->SetGroups(1);

        // muDNN convolution configuration
        conv_operator->SetNdInfo(
            static_cast<int>(info.ndim()),
            pad_dims.data(),
            stride_dims.data(),
            dilation_dims.data()
        );

     
        // 10. Select algorithm (simple version: always query)
        ::musa::dnn::Convolution::Algorithm algo;
        conv_operator->GetRecommendForwardAlgorithm(
            mudnn_handle,
            algo,
            output_tensor,
            input_tensor,
            weight_tensor
        );

        // 11. Workspace memory handler
        ::musa::dnn::MemoryMaintainer maintainer =
            [](size_t size) -> ::musa::dnn::MemoryHandler {
                void* ptr = nullptr;
                musaMalloc(&ptr, size);
                return ::musa::dnn::MemoryHandler(
                    ptr,
                    [](void* p) { if (p) musaFree(p); }
                );
            };

        // 12. Run convolution (no fused activation)
        ::musa::dnn::Tensor add_tensor;  // unused
        ::musa::dnn::Convolution::FusedActivationDesc act;
        act.SetMode(::musa::dnn::Convolution::FusedActivationDesc::Mode::IDENTITY);

        conv_operator->RunFusion(
            mudnn_handle,
            output_tensor,
            input_tensor,
            weight_tensor,
            bias != nullptr ? bias_tensor : ::musa::dnn::Tensor(),
            add_tensor,
            act,
            algo,
            maintainer
        );

        return INFINI_STATUS_SUCCESS;
    });
}


infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    const void *bias,
    void *stream) const {


    // Check for null pointers
    if (!_opaque) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (!_opaque->internal) {
        return INFINI_STATUS_BAD_PARAM;
    }

    switch (_dtype) {
        case INFINI_DTYPE_F16:
            return mudnn::calculate<half>(_info, _opaque->internal, y, x, w, bias, stream);
        case INFINI_DTYPE_F32:
            return mudnn::calculate<float>(_info, _opaque->internal, y, x, w, bias, stream);
        case INFINI_DTYPE_BF16:
            return mudnn::calculate<__mt_bfloat16>(_info, _opaque->internal, y, x, w, bias, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::conv::mudnn