#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

namespace infinicore {

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;
    if (exponent == 31) {
        if (mantissa != 0) {
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
            f32 = sign | 0x7F800000;
        }
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

inline float bf16_to_f32(uint16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val) << 16;
    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

// Template function for printing data recursively
template <typename T>
void print_data(const T *data, const Shape &shape, const Strides &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << data[i * strides[dim]] << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

// Specialization for F16 (uint16_t)
template <>
void print_data<uint16_t>(const uint16_t *data, const Shape &shape, const Strides &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << f16_to_f32(data[i * strides[dim]]) << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

// Function for printing BF16 data
void print_data_bf16(const uint16_t *data, const Shape &shape, const Strides &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            std::cout << bf16_to_f32(data[i * strides[dim]]) << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data_bf16(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

// Template function for writing data recursively to binary file (handles non-contiguous tensors)
template <typename T>
void write_binary_data(std::ofstream &out, const T *data, const Shape &shape, const Strides &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        // Write the innermost dimension
        for (size_t i = 0; i < shape[dim]; i++) {
            out.write(reinterpret_cast<const char *>(&data[i * strides[dim]]), sizeof(T));
        }
    } else {
        // Recursively process higher dimensions
        for (size_t i = 0; i < shape[dim]; i++) {
            write_binary_data(out, data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void TensorImpl::debug(const std::string &filename) const {
    // Synchronize device if needed
    context::syncDevice();

    std::cout << info() << std::endl;

    const std::byte *cpu_data = nullptr;
    std::unique_ptr<std::byte[]> allocated_memory; // RAII: 自动管理内存

    // Copy data to CPU if not already on CPU
    if (this->device().getType() != Device::Type::CPU) {
        size_t numel = this->numel();
        size_t element_size = dsize(this->dtype());

        // 检查乘法溢出
        if (numel > 0 && element_size > std::numeric_limits<size_t>::max() / numel) {
            std::cerr << "Error: Memory size calculation overflow for tensor with "
                      << numel << " elements of size " << element_size << "\n";
            return;
        }

        size_t mem_size = numel * element_size;
        allocated_memory = std::make_unique<std::byte[]>(mem_size);
        context::memcpyD2H(allocated_memory.get(), this->data(), mem_size);
        cpu_data = allocated_memory.get();
    } else {
        cpu_data = this->data();
    }

    // If filename is provided, save to binary file
    if (!filename.empty()) {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error opening file for writing: " << filename << "\n";
            return; // allocated_memory 会自动释放（RAII）
        }

        // Check if tensor is contiguous - for optimization
        if (this->is_contiguous()) {
            // Fast path: contiguous tensor, write in one go
            size_t mem_size = this->numel() * dsize(this->dtype());
            outFile.write(reinterpret_cast<const char *>(cpu_data), mem_size);
        } else {
            // Slow path: non-contiguous tensor, write element by element using strides
            switch (this->dtype()) {
            case DataType::F16:
            case DataType::BF16:
                write_binary_data(outFile, reinterpret_cast<const uint16_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::F32:
                write_binary_data(outFile, reinterpret_cast<const float *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::F64:
                write_binary_data(outFile, reinterpret_cast<const double *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::U64:
                write_binary_data(outFile, reinterpret_cast<const uint64_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::I64:
                write_binary_data(outFile, reinterpret_cast<const int64_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::U32:
                write_binary_data(outFile, reinterpret_cast<const uint32_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::I32:
                write_binary_data(outFile, reinterpret_cast<const int32_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::U16:
                write_binary_data(outFile, reinterpret_cast<const uint16_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::I16:
                write_binary_data(outFile, reinterpret_cast<const int16_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::U8:
                write_binary_data(outFile, reinterpret_cast<const uint8_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::I8:
                write_binary_data(outFile, reinterpret_cast<const int8_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            case DataType::BOOL:
                // 布尔类型特殊处理：转换为 uint8_t 以保证跨平台一致性
                write_binary_data(outFile, reinterpret_cast<const uint8_t *>(cpu_data),
                                  this->shape(), this->strides(), 0);
                break;
            default:
                std::cerr << "Unsupported data type for binary output\n";
                return;
            }
        }

        // 显式关闭文件并检查是否成功
        outFile.close();
        if (!outFile) {
            std::cerr << "Error: Failed to write data to file: " << filename << "\n";
            return;
        }

        std::cout << "Data written to binary file: " << filename;
        if (!this->is_contiguous()) {
            std::cout << " (non-contiguous tensor, wrote " << this->numel() << " elements)";
        }
        std::cout << "\n";
        return;
    }

    // Print data based on dtype
    switch (this->dtype()) {
    case DataType::F16:
        print_data(reinterpret_cast<const uint16_t *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::F32:
        print_data(reinterpret_cast<const float *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::F64:
        print_data(reinterpret_cast<const double *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::U64:
        print_data(reinterpret_cast<const uint64_t *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::I64:
        print_data(reinterpret_cast<const int64_t *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::U32:
        print_data(reinterpret_cast<const uint32_t *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::I32:
        print_data(reinterpret_cast<const int32_t *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::U16:
        print_data(reinterpret_cast<const uint16_t *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::I16:
        print_data(reinterpret_cast<const int16_t *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::U8:
        print_data(reinterpret_cast<const uint8_t *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::I8:
        print_data(reinterpret_cast<const int8_t *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    case DataType::BF16:
        print_data_bf16(reinterpret_cast<const uint16_t *>(cpu_data),
                        this->shape(), this->strides(), 0);
        break;
    case DataType::BOOL:
        print_data(reinterpret_cast<const bool *>(cpu_data),
                   this->shape(), this->strides(), 0);
        break;
    default:
        std::cout << "Unsupported data type for debug" << std::endl;
        break;
    }
}

void TensorImpl::debug() const {
    this->debug("");
}

} // namespace infinicore
