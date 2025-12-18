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

    // Debug: Print input parameters
    printf("DEBUG: conv_mudnn create called with handle_=%p, n=%zu\n", (void*)handle_, n);
    if (y_desc) printf("DEBUG: y_desc dims=");
    if (x_desc) printf("DEBUG: x_desc dims=");
    if (w_desc) printf("DEBUG: w_desc dims=");
    fflush(stdout);

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc, b_desc, pads, strides, dilations, n);
    CHECK_RESULT(result);

    auto info = result.take();

    printf("DEBUG: Creating descriptor with batch=%zu, in_channels=%zu, out_channels=%zu, ndim=%zu\n",
           info.batch(), info.in_channels(), info.out_channels(), info.ndim());
    fflush(stdout);

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

    printf("DEBUG: conv_mudnn calculate called with info batch=%zu\n", info.batch());
    printf("DEBUG: Pointers - y=%p, x=%p, w=%p, bias=%p, stream=%p\n", y, x, w, bias, stream);
    fflush(stdout);

    // Use muDNN handle management
    return _internal->useMudnn((musaStream_t)stream, [&](::musa::dnn::Handle &mudnn_handle) -> infiniStatus_t {

        printf("DEBUG: Inside muDNN lambda\n");
        printf("DEBUG: About to create conv_operator\n");
        fflush(stdout);

        // Create convolution operator
        auto conv_operator = std::make_unique<::musa::dnn::Convolution>();

        printf("DEBUG: conv_operator created successfully\n");
        fflush(stdout);

        conv_operator->SetComputeMode(::musa::dnn::Convolution::ComputeMode::TENSOR);

        printf("DEBUG: SetComputeMode done\n");
        fflush(stdout);

        // Set tensor data types
        ::musa::dnn::Tensor::Type tensor_type;
        if constexpr (std::is_same<Tdata, half>::value) {
            tensor_type = ::musa::dnn::Tensor::Type::HALF;
        } else if constexpr (std::is_same<Tdata, __mt_bfloat16>::value) {
            tensor_type = ::musa::dnn::Tensor::Type::BFLOAT16;
        } else {
            tensor_type = ::musa::dnn::Tensor::Type::FLOAT;
        }

printf("1111\n");
        fflush(stdout);

        // Create tensors
        ::musa::dnn::Tensor input_tensor, output_tensor, weight_tensor, bias_tensor;

        printf("DEBUG: About to configure input_tensor\n");
        fflush(stdout);

        // Configure input tensor [N, C, H, W, ...]
        input_tensor.SetType(tensor_type);

        printf("DEBUG: SetType done, about to SetFormat\n");
        fflush(stdout);

        input_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);

        printf("DEBUG: SetFormat done, about to create input_dims\n");
        fflush(stdout);

        std::vector<int64_t> input_dims = {
            static_cast<int64_t>(info.batch()),
            static_cast<int64_t>(info.in_channels())
        };

        printf("DEBUG: Basic input_dims: batch=%ld, in_channels=%ld\n",
               input_dims[0], input_dims[1]);
        fflush(stdout);

        for (size_t i = 0; i < info.ndim(); ++i) {
            input_dims.push_back(static_cast<int64_t>(info.input_dim(i)));
            printf("DEBUG: input_dim[%zu]=%zu\n", i, info.input_dim(i));
            fflush(stdout);
        }

        printf("DEBUG: About to SetNdInfo for input_tensor\n");
        fflush(stdout);

        // Calculate strides like GEMM does
        std::vector<int64_t> input_strides(input_dims.size());
        input_strides[input_dims.size() - 1] = 1;  // Innermost dimension has stride 1

        // Calculate strides for other dimensions (row-major)
        for (int i = input_dims.size() - 2; i >= 0; --i) {
            input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
        }

        printf("DEBUG: input_strides calculated\n");
        fflush(stdout);

        input_tensor.SetNdInfo(static_cast<int>(input_dims.size()), input_dims.data(), input_strides.data());

        printf("DEBUG: SetNdInfo done, about to SetAddr\n");
        fflush(stdout);

        input_tensor.SetAddr(const_cast<void*>(x));

        printf("DEBUG: input_tensor configuration done\n");
        fflush(stdout);

printf("2222\n");
        fflush(stdout);

        // Configure output tensor [N, K, H_out, W_out, ...]
        output_tensor.SetType(tensor_type);
        output_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
        std::vector<int64_t> output_dims = {
            static_cast<int64_t>(info.batch()),
            static_cast<int64_t>(info.out_channels())
        };
        for (size_t i = 0; i < info.ndim(); ++i) {
            output_dims.push_back(static_cast<int64_t>(info.output_dim(i)));
        }

        // Calculate strides for output tensor
        std::vector<int64_t> output_strides(output_dims.size());
        output_strides[output_dims.size() - 1] = 1;
        for (int i = output_dims.size() - 2; i >= 0; --i) {
            output_strides[i] = output_strides[i + 1] * output_dims[i + 1];
        }

        output_tensor.SetNdInfo(static_cast<int>(output_dims.size()), output_dims.data(), output_strides.data());
        output_tensor.SetAddr(y);

printf("3333\n");
        fflush(stdout);

        // Configure weight tensor [K, C, H_k, W_k, ...]
        weight_tensor.SetType(tensor_type);
        weight_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);
        std::vector<int64_t> weight_dims = {
            static_cast<int64_t>(info.out_channels()),
            static_cast<int64_t>(info.in_channels())
        };

printf("4444\n");
        fflush(stdout);

        for (size_t i = 0; i < info.ndim(); ++i) {
            weight_dims.push_back(static_cast<int64_t>(info.kernel_dim(i)));
        }

        // Calculate strides for weight tensor
        std::vector<int64_t> weight_strides(weight_dims.size());
        weight_strides[weight_dims.size() - 1] = 1;
        for (int i = weight_dims.size() - 2; i >= 0; --i) {
            weight_strides[i] = weight_strides[i + 1] * weight_dims[i + 1];
        }

        weight_tensor.SetNdInfo(static_cast<int>(weight_dims.size()), weight_dims.data(), weight_strides.data());
        weight_tensor.SetAddr(const_cast<void*>(w));

printf("5555\n");
fflush(stdout);


        // Configure bias tensor if provided
        if (bias != nullptr) {
            bias_tensor.SetType(tensor_type);
            bias_tensor.SetFormat(::musa::dnn::Tensor::Format::NCHW);

            // For convolution bias, it should be a 1D tensor [out_channels]
            std::vector<int64_t> bias_dims = {
                static_cast<int64_t>(info.out_channels())
            };

            // For 1D bias tensor, stride is simply [1]
            std::vector<int64_t> bias_strides = {1};

            bias_tensor.SetNdInfo(static_cast<int>(bias_dims.size()), bias_dims.data(), bias_strides.data());
            bias_tensor.SetAddr(const_cast<void*>(bias));
        }


printf("6666\n");
fflush(stdout);



        // Set convolution parameters
        std::vector<int> pad_dims(info.ndim());
        std::vector<int> stride_dims(info.ndim());
        std::vector<int> dilation_dims(info.ndim());

        for (size_t i = 0; i < info.ndim(); ++i) {
            pad_dims[i] = static_cast<int>(info.pad_info(i));
            stride_dims[i] = static_cast<int>(info.stride_info(i));
            dilation_dims[i] = static_cast<int>(info.dilation_info(i));
        }



printf("7777\n");
fflush(stdout);



        conv_operator->SetGroups(1);  // Default to groups = 1
        conv_operator->SetNdInfo(info.ndim(), pad_dims.data(), stride_dims.data(), dilation_dims.data());

        // Get recommended algorithm
        ::musa::dnn::Convolution::Algorithm algo;
        conv_operator->GetRecommendForwardAlgorithm(mudnn_handle, algo, output_tensor, input_tensor, weight_tensor);

printf("8888\n");
fflush(stdout);


        // Workspace memory handler
        ::musa::dnn::MemoryMaintainer maintainer = [](size_t size) -> ::musa::dnn::MemoryHandler {
            void* ptr = nullptr;
            musaMalloc(&ptr, size);
            return ::musa::dnn::MemoryHandler(ptr, [](void* p) { if(p) musaFree(p); });
        };

printf("9999\n");
fflush(stdout);


        // Create empty activation (identity)
        ::musa::dnn::Convolution::FusedActivationDesc act_desc;
        act_desc.SetMode(::musa::dnn::Convolution::FusedActivationDesc::Mode::IDENTITY);

        // Run convolution
        if (bias != nullptr) {

printf("10\n");
fflush(stdout);


            // Run with bias using RunFusion
            conv_operator->RunFusion(
                mudnn_handle,
                output_tensor,
                input_tensor,
                weight_tensor,
                bias_tensor,
                ::musa::dnn::Tensor(),  // add tensor (empty)
                act_desc,
                algo,
                maintainer
            );
        } else {

printf("11\n");
fflush(stdout);


            // Run without bias using standard Run
            conv_operator->Run(
                mudnn_handle,
                output_tensor,
                input_tensor,
                weight_tensor,
                algo,
                maintainer
            );
        }

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

    printf("DEBUG: Descriptor::calculate called\n");
    fflush(stdout);

    // Check for null pointers
    if (!_opaque) {
        printf("ERROR: _opaque is null!\n");
        fflush(stdout);
        return INFINI_STATUS_BAD_PARAM;
    }
    if (!_opaque->internal) {
        printf("ERROR: _opaque->internal is null!\n");
        fflush(stdout);
        return INFINI_STATUS_BAD_PARAM;
    }

    switch (_dtype) {
        case INFINI_DTYPE_F16:
            printf("DEBUG: Calling mudnn::calculate<half>\n");
            fflush(stdout);
            return mudnn::calculate<half>(_info, _opaque->internal, y, x, w, bias, stream);
        case INFINI_DTYPE_F32:
            printf("DEBUG: Calling mudnn::calculate<float>\n");
            fflush(stdout);
            return mudnn::calculate<float>(_info, _opaque->internal, y, x, w, bias, stream);
        case INFINI_DTYPE_BF16:
            printf("DEBUG: Calling mudnn::calculate<__mt_bfloat16>\n");
            fflush(stdout);
            return mudnn::calculate<__mt_bfloat16>(_info, _opaque->internal, y, x, w, bias, stream);
        default:
            printf("ERROR: Unsupported dtype: %d\n", _dtype);
            fflush(stdout);
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::conv::mudnn