#include <infinicore.hpp>

#include <cstring>
#include <cstdint>
#include <stdexcept>

namespace infinicore {

std::string toString(const DataType &dtype) {
    switch (dtype) {
    case DataType::BYTE:
        return "BYTE";
    case DataType::BOOL:
        return "BOOL";
    case DataType::I8:
        return "I8";
    case DataType::I16:
        return "I16";
    case DataType::I32:
        return "I32";
    case DataType::I64:
        return "I64";
    case DataType::U8:
        return "U8";
    case DataType::U16:
        return "U16";
    case DataType::U32:
        return "U32";
    case DataType::U64:
        return "U64";
    case DataType::F8:
        return "F8";
    case DataType::F16:
        return "F16";
    case DataType::F32:
        return "F32";
    case DataType::F64:
        return "F64";
    case DataType::C16:
        return "C16";
    case DataType::C32:
        return "C32";
    case DataType::C64:
        return "C64";
    case DataType::C128:
        return "C128";
    case DataType::BF16:
        return "BF16";
    }

    // TODO: Add error handling.
    return "";
}

size_t dsize(const DataType &dtype) {
    switch (dtype) {
    case DataType::BYTE:
    case DataType::BOOL:
    case DataType::F8:
    case DataType::I8:
    case DataType::U8:
        return 1;
    case DataType::I16:
    case DataType::U16:
    case DataType::F16:
    case DataType::BF16:
    case DataType::C16:
        return 2;
    case DataType::I32:
    case DataType::U32:
    case DataType::F32:
    case DataType::C32:
        return 4;
    case DataType::I64:
    case DataType::U64:
    case DataType::F64:
    case DataType::C64:
        return 8;
    case DataType::C128:
        return 16;
    }

    // TODO: Add error handling.
    return 0;
}

void convertFloat(double value, DataType dtype, void* buffer) {
    switch (dtype){
        case DataType::F32: {
            float f32_val = static_cast<float>(value);
            std::memcpy(buffer, &f32_val, sizeof(float));
            break;
        }
        case DataType::F64: {
            double f64_val = value;
            std::memcpy(buffer, &f64_val, sizeof(double));
            break;
        }
        case DataType::F16: {
            float f32_val = static_cast<float>(value);
            uint32_t f;
            std::memcpy(&f, &f32_val, sizeof(float));
            
            uint16_t h;
            uint32_t sign = (f >> 16) & 0x8000;
            int32_t exp = ((f >> 23) & 0xff) - 127 + 15;
            uint32_t mant = f & 0x7fffff;
            if (exp <= 0) {
                h = sign;
            } else if (exp >= 31) {
                h = sign | 0x7c00;
            } else {
                h = sign | (exp << 10) | (mant >> 13);
            }
            std::memcpy(buffer, &h, sizeof(uint16_t));
            break;
        }
        case DataType::BF16: {
            float f32_val = static_cast<float>(value);
            uint32_t f;
            std::memcpy(&f, &f32_val, sizeof(float));
            uint16_t bf16 = static_cast<uint16_t>(f >> 16);
            std::memcpy(buffer, &bf16, sizeof(uint16_t));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for float conversion");
    }
}
} // namespace infinicore
