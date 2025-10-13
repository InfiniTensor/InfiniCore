#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
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

void TensorImpl::debug(const std::string &filename) const {
    // Synchronize device if needed
    context::syncDevice();

    std::cout << info() << std::endl;

    const std::byte *cpu_data = nullptr;
    std::byte *allocated_memory = nullptr;

    // Copy data to CPU if not already on CPU
    if (this->device().getType() != Device::Type::CPU) {
        size_t mem_size = this->numel() * dsize(this->dtype());
        allocated_memory = new std::byte[mem_size];
        context::memcpyD2H(allocated_memory, this->data(), mem_size);
        cpu_data = allocated_memory;
    } else {
        cpu_data = this->data();
    }

    // If filename is provided, save to file
    if (!filename.empty()) {
        // Determine file format based on extension
        bool is_text_format = false;
        size_t dot_pos = filename.find_last_of('.');
        if (dot_pos != std::string::npos) {
            std::string ext = filename.substr(dot_pos);
            is_text_format = (ext == ".txt");
        }

        if (is_text_format) {
            // Save as text format
            std::ofstream outFile(filename);
            if (!outFile) {
                std::cerr << "Error opening file for writing: " << filename << "\n";
                if (allocated_memory) {
                    delete[] allocated_memory;
                }
                return;
            }

            // Write header with tensor information
            outFile << "# Tensor Debug Output\n";
            outFile << "# Shape: [";
            for (size_t i = 0; i < this->shape().size(); ++i) {
                outFile << this->shape()[i];
                if (i < this->shape().size() - 1) {
                    outFile << ", ";
                }
            }
            outFile << "]\n";
            outFile << "# Strides: [";
            for (size_t i = 0; i < this->strides().size(); ++i) {
                outFile << this->strides()[i];
                if (i < this->strides().size() - 1) {
                    outFile << ", ";
                }
            }
            outFile << "]\n";
            outFile << "# Dtype: " << toString(this->dtype()) << "\n";
            outFile << "# Contiguous: " << (this->is_contiguous() ? "Yes" : "No") << "\n";
            outFile << "# Elements: " << this->numel() << "\n";
            outFile << "#\n";

            // Helper function to write data recursively
            std::function<void(const std::byte *, const Shape &, const Strides &, size_t, std::ofstream &)> write_data;

            switch (this->dtype()) {
            case DataType::F16:
                write_data = [&write_data](const std::byte *data, const Shape &shape, const Strides &strides, size_t dim, std::ofstream &out) {
                    const uint16_t *ptr = reinterpret_cast<const uint16_t *>(data);
                    if (dim == shape.size() - 1) {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            out << f16_to_f32(ptr[i * strides[dim]]);
                            if (i < shape[dim] - 1) {
                                out << " ";
                            }
                        }
                        out << "\n";
                    } else {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            write_data(data + i * strides[dim] * sizeof(uint16_t), shape, strides, dim + 1, out);
                        }
                    }
                };
                break;
            case DataType::F32:
                write_data = [&write_data](const std::byte *data, const Shape &shape, const Strides &strides, size_t dim, std::ofstream &out) {
                    const float *ptr = reinterpret_cast<const float *>(data);
                    if (dim == shape.size() - 1) {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            out << ptr[i * strides[dim]];
                            if (i < shape[dim] - 1) {
                                out << " ";
                            }
                        }
                        out << "\n";
                    } else {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            write_data(data + i * strides[dim] * sizeof(float), shape, strides, dim + 1, out);
                        }
                    }
                };
                break;
            case DataType::F64:
                write_data = [&write_data](const std::byte *data, const Shape &shape, const Strides &strides, size_t dim, std::ofstream &out) {
                    const double *ptr = reinterpret_cast<const double *>(data);
                    if (dim == shape.size() - 1) {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            out << ptr[i * strides[dim]];
                            if (i < shape[dim] - 1) {
                                out << " ";
                            }
                        }
                        out << "\n";
                    } else {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            write_data(data + i * strides[dim] * sizeof(double), shape, strides, dim + 1, out);
                        }
                    }
                };
                break;
            case DataType::I32:
                write_data = [&write_data](const std::byte *data, const Shape &shape, const Strides &strides, size_t dim, std::ofstream &out) {
                    const int32_t *ptr = reinterpret_cast<const int32_t *>(data);
                    if (dim == shape.size() - 1) {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            out << ptr[i * strides[dim]];
                            if (i < shape[dim] - 1) {
                                out << " ";
                            }
                        }
                        out << "\n";
                    } else {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            write_data(data + i * strides[dim] * sizeof(int32_t), shape, strides, dim + 1, out);
                        }
                    }
                };
                break;
            case DataType::I64:
                write_data = [&write_data](const std::byte *data, const Shape &shape, const Strides &strides, size_t dim, std::ofstream &out) {
                    const int64_t *ptr = reinterpret_cast<const int64_t *>(data);
                    if (dim == shape.size() - 1) {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            out << ptr[i * strides[dim]];
                            if (i < shape[dim] - 1) {
                                out << " ";
                            }
                        }
                        out << "\n";
                    } else {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            write_data(data + i * strides[dim] * sizeof(int64_t), shape, strides, dim + 1, out);
                        }
                    }
                };
                break;
            case DataType::BF16:
                write_data = [&write_data](const std::byte *data, const Shape &shape, const Strides &strides, size_t dim, std::ofstream &out) {
                    const uint16_t *ptr = reinterpret_cast<const uint16_t *>(data);
                    if (dim == shape.size() - 1) {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            out << bf16_to_f32(ptr[i * strides[dim]]);
                            if (i < shape[dim] - 1) {
                                out << " ";
                            }
                        }
                        out << "\n";
                    } else {
                        for (size_t i = 0; i < shape[dim]; i++) {
                            write_data(data + i * strides[dim] * sizeof(uint16_t), shape, strides, dim + 1, out);
                        }
                    }
                };
                break;
            default:
                outFile << "# Unsupported data type for text output\n";
                outFile.close();
                if (allocated_memory) {
                    delete[] allocated_memory;
                }
                return;
            }

            // Write the actual data
            write_data(cpu_data, this->shape(), this->strides(), 0, outFile);

            outFile.close();
            std::cout << "Data written to text file: " << filename << "\n";
        } else {
            // Save as binary format (default)
            std::ofstream outFile(filename, std::ios::binary);
            if (!outFile) {
                std::cerr << "Error opening file for writing: " << filename << "\n";
                if (allocated_memory) {
                    delete[] allocated_memory;
                }
                return;
            }
            size_t mem_size = this->numel() * dsize(this->dtype());
            outFile.write(reinterpret_cast<const char *>(cpu_data), mem_size);
            outFile.close();
            std::cout << "Data written to binary file: " << filename << "\n";
        }

        if (allocated_memory) {
            delete[] allocated_memory;
        }
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

    // Clean up allocated memory
    if (allocated_memory) {
        delete[] allocated_memory;
    }
}

void TensorImpl::debug() const {
    this->debug("");
}

} // namespace infinicore
