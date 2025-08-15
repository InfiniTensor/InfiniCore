#include "cast_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../../../../utils/custom_types.h"

namespace op::cast::cpu {

struct Descriptor::Opaque {
    size_t numel;
};

Descriptor::Descriptor(infiniDtype_t input_dtype, infiniDtype_t output_dtype)
    : InfiniopDescriptor{INFINI_DEVICE_CPU, 0}, _input_dtype(input_dtype), _output_dtype(output_dtype) {
    _opaque = new Opaque();
}

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    // auto handle = reinterpret_cast<device::cpu::Handle *>(handle_); // 暂时注释掉未使用的变量
    auto input_dtype = input_desc_vec[0]->dtype();
    auto output_dtype = output_desc->dtype();

    // 检查支持的类型转换
    bool valid_cast = false;
    
    // 整数类型之间的转换（包括uint8）
    if ((input_dtype == INFINI_DTYPE_I32 || input_dtype == INFINI_DTYPE_I64 || 
         input_dtype == INFINI_DTYPE_U32 || input_dtype == INFINI_DTYPE_U64 || input_dtype == INFINI_DTYPE_U8) &&
        (output_dtype == INFINI_DTYPE_I32 || output_dtype == INFINI_DTYPE_I64 || 
         output_dtype == INFINI_DTYPE_U32 || output_dtype == INFINI_DTYPE_U64 || output_dtype == INFINI_DTYPE_U8)) {
        valid_cast = true;
    }
    
    // 浮点类型之间的转换
    if ((input_dtype == INFINI_DTYPE_F64 || input_dtype == INFINI_DTYPE_F32 || input_dtype == INFINI_DTYPE_F16) &&
        (output_dtype == INFINI_DTYPE_F64 || output_dtype == INFINI_DTYPE_F32 || output_dtype == INFINI_DTYPE_F16)) {
        valid_cast = true;
    }
    
    // 整数类型转浮点类型（包括uint8）
    if ((input_dtype == INFINI_DTYPE_I32 || input_dtype == INFINI_DTYPE_I64 || 
         input_dtype == INFINI_DTYPE_U32 || input_dtype == INFINI_DTYPE_U64 || input_dtype == INFINI_DTYPE_U8) &&
        (output_dtype == INFINI_DTYPE_F64 || output_dtype == INFINI_DTYPE_F32 || output_dtype == INFINI_DTYPE_F16)) {
        valid_cast = true;
    }
    
    // 浮点类型转整数类型（包括uint8）
    if ((input_dtype == INFINI_DTYPE_F64 || input_dtype == INFINI_DTYPE_F32 || input_dtype == INFINI_DTYPE_F16) &&
        (output_dtype == INFINI_DTYPE_I32 || output_dtype == INFINI_DTYPE_I64 || 
         output_dtype == INFINI_DTYPE_U32 || output_dtype == INFINI_DTYPE_U64 || output_dtype == INFINI_DTYPE_U8)) {
        valid_cast = true;
    }
    
    if (!valid_cast) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // 检查形状一致性
    const auto &input_shape = input_desc_vec[0]->shape();
    const auto &output_shape = output_desc->shape();
    if (input_shape != output_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto desc = new Descriptor(input_dtype, output_dtype);
    desc->_opaque->numel = output_desc->numel();
    
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::workspaceSize() const {
    return 0;
}

// 类型转换辅助函数模板
template<typename InputType, typename OutputType>
void cast_elements(const InputType* input, OutputType* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = utils::cast<OutputType>(input[i]);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    const void *input = inputs[0];
    size_t numel = _opaque->numel;

    // 根据输入和输出数据类型进行转换
    if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_I64) {
        cast_elements<int32_t, int64_t>(static_cast<const int32_t*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_I32) {
        cast_elements<int64_t, int32_t>(static_cast<const int64_t*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_F64) {
        cast_elements<float, double>(static_cast<const float*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_F32) {
        cast_elements<double, float>(static_cast<const double*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_F32) {
        cast_elements<fp16_t, float>(static_cast<const fp16_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_F16) {
        cast_elements<float, fp16_t>(static_cast<const float*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_F64) {
        cast_elements<fp16_t, double>(static_cast<const fp16_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_F16) {
        cast_elements<double, fp16_t>(static_cast<const double*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_F32) {
        cast_elements<int32_t, float>(static_cast<const int32_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_F64) {
        cast_elements<int32_t, double>(static_cast<const int32_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_F16) {
        cast_elements<int32_t, fp16_t>(static_cast<const int32_t*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_F32) {
        cast_elements<int64_t, float>(static_cast<const int64_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_F64) {
        cast_elements<int64_t, double>(static_cast<const int64_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_F16) {
        cast_elements<int64_t, fp16_t>(static_cast<const int64_t*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_I32) {
        cast_elements<float, int32_t>(static_cast<const float*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_I64) {
        cast_elements<float, int64_t>(static_cast<const float*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_I32) {
        cast_elements<double, int32_t>(static_cast<const double*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_I64) {
        cast_elements<double, int64_t>(static_cast<const double*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_I32) {
        cast_elements<fp16_t, int32_t>(static_cast<const fp16_t*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_I64) {
        cast_elements<fp16_t, int64_t>(static_cast<const fp16_t*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_U64) {
        cast_elements<uint32_t, uint64_t>(static_cast<const uint32_t*>(input), static_cast<uint64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_U32) {
        cast_elements<uint64_t, uint32_t>(static_cast<const uint64_t*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_U32) {
        cast_elements<int32_t, uint32_t>(static_cast<const int32_t*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_I32) {
        cast_elements<uint32_t, int32_t>(static_cast<const uint32_t*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_U64) {
        cast_elements<int64_t, uint64_t>(static_cast<const int64_t*>(input), static_cast<uint64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_I64) {
        cast_elements<uint64_t, int64_t>(static_cast<const uint64_t*>(input), static_cast<int64_t*>(output), numel);
    }
    // 无符号整数到浮点类型的转换
    else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_F32) {
        cast_elements<uint32_t, float>(static_cast<const uint32_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_F64) {
        cast_elements<uint32_t, double>(static_cast<const uint32_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_F16) {
        cast_elements<uint32_t, fp16_t>(static_cast<const uint32_t*>(input), static_cast<fp16_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_F32) {
        cast_elements<uint64_t, float>(static_cast<const uint64_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_F64) {
        cast_elements<uint64_t, double>(static_cast<const uint64_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_F16) {
        cast_elements<uint64_t, fp16_t>(static_cast<const uint64_t*>(input), static_cast<fp16_t*>(output), numel);
    }
    // 浮点类型到无符号整数类型的转换
    else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_U32) {
        cast_elements<float, uint32_t>(static_cast<const float*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_U64) {
        cast_elements<float, uint64_t>(static_cast<const float*>(input), static_cast<uint64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_U32) {
        cast_elements<double, uint32_t>(static_cast<const double*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_U64) {
        cast_elements<double, uint64_t>(static_cast<const double*>(input), static_cast<uint64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_U32) {
        cast_elements<fp16_t, uint32_t>(static_cast<const fp16_t*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_U64) {
        cast_elements<fp16_t, uint64_t>(static_cast<const fp16_t*>(input), static_cast<uint64_t*>(output), numel);
    }
    // uint8类型的转换支持
    else if (_input_dtype == INFINI_DTYPE_U8 && _output_dtype == INFINI_DTYPE_U32) {
        cast_elements<uint8_t, uint32_t>(static_cast<const uint8_t*>(input), static_cast<uint32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U8 && _output_dtype == INFINI_DTYPE_U64) {
        cast_elements<uint8_t, uint64_t>(static_cast<const uint8_t*>(input), static_cast<uint64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U8 && _output_dtype == INFINI_DTYPE_I32) {
        cast_elements<uint8_t, int32_t>(static_cast<const uint8_t*>(input), static_cast<int32_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U8 && _output_dtype == INFINI_DTYPE_I64) {
        cast_elements<uint8_t, int64_t>(static_cast<const uint8_t*>(input), static_cast<int64_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U8 && _output_dtype == INFINI_DTYPE_F32) {
        cast_elements<uint8_t, float>(static_cast<const uint8_t*>(input), static_cast<float*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U8 && _output_dtype == INFINI_DTYPE_F64) {
        cast_elements<uint8_t, double>(static_cast<const uint8_t*>(input), static_cast<double*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U8 && _output_dtype == INFINI_DTYPE_F16) {
        cast_elements<uint8_t, fp16_t>(static_cast<const uint8_t*>(input), static_cast<fp16_t*>(output), numel);
    }
    // 其他类型到uint8的转换
    else if (_input_dtype == INFINI_DTYPE_U32 && _output_dtype == INFINI_DTYPE_U8) {
        cast_elements<uint32_t, uint8_t>(static_cast<const uint32_t*>(input), static_cast<uint8_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_U64 && _output_dtype == INFINI_DTYPE_U8) {
        cast_elements<uint64_t, uint8_t>(static_cast<const uint64_t*>(input), static_cast<uint8_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I32 && _output_dtype == INFINI_DTYPE_U8) {
        cast_elements<int32_t, uint8_t>(static_cast<const int32_t*>(input), static_cast<uint8_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_I64 && _output_dtype == INFINI_DTYPE_U8) {
        cast_elements<int64_t, uint8_t>(static_cast<const int64_t*>(input), static_cast<uint8_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F32 && _output_dtype == INFINI_DTYPE_U8) {
        cast_elements<float, uint8_t>(static_cast<const float*>(input), static_cast<uint8_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F64 && _output_dtype == INFINI_DTYPE_U8) {
        cast_elements<double, uint8_t>(static_cast<const double*>(input), static_cast<uint8_t*>(output), numel);
    } else if (_input_dtype == INFINI_DTYPE_F16 && _output_dtype == INFINI_DTYPE_U8) {
        cast_elements<fp16_t, uint8_t>(static_cast<const fp16_t*>(input), static_cast<uint8_t*>(output), numel);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}



} // namespace op::cast::cpu