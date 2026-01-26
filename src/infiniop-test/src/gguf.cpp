#include "gguf.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

std::string GGUFKeyValue::toString() const {
    std::ostringstream oss;
    oss << "Key: " << key << ", Type: " << GGUF_TYPE_NAME[gguf_type] << ", Value: ";
    if (gguf_type == GGUF_TYPE_STRING) {
        std::string str(value.begin(), value.end());
        oss << str;
    } else if (value.size() > GGUF_TYPE_SIZE[gguf_type]) {
        oss << "[";
        for (size_t i = 0; i < value.size() / GGUF_TYPE_SIZE[gguf_type]; ++i) {
            oss << ggufDataToString(value.data() + i * GGUF_TYPE_SIZE[gguf_type], gguf_type);
            if (i < value.size() / GGUF_TYPE_SIZE[gguf_type] - 1) {
                oss << ", ";
            }
        }
        oss << "]";
    } else {
        oss << ggufDataToString(value.data(), gguf_type);
    }

    return oss.str();
}

std::string GGUFTensorInfo::toString() const {
    std::ostringstream oss;
    oss << "Name: " << name << ", NDims: " << ndim << ", Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i < shape.size() - 1) {
            oss << ", ";
        }
    }
    oss << "], DataType: " << GGML_TYPE_NAME[ggml_type] << ", DataOffset: " << data_offset;
    return oss.str();
}

GGUFFileReader::GGUFFileReader(const std::string &filepath) {
    try {
        _file = std::make_shared<FileMapping>(filepath);
    } catch (const std::exception &e) {
        throw e;
    }
    _data = _file->ptr();
    _cursor = reinterpret_cast<uint8_t *>(_data);
    readHeader();
    readMetaKVs();
    readTensorInfos();
    size_t padding = (size_t)(32 - ((char *)_cursor - (char *)_data) % 32) % 32;
    _cursor += padding;
    _data_start = _cursor;  // Mark the start of tensor data section
}

const std::unordered_map<std::string, std::shared_ptr<GGUFKeyValue>> &
GGUFFileReader::getAttributeMap() const {
    return _attributes_map;
}

const std::unordered_map<std::string, std::shared_ptr<GGUFTensorInfo>> &
GGUFFileReader::getTensorInfoMap() const {
    return _tensors_info_map;
}

void GGUFFileReader::readHeader() {
    if (std::memcmp(_cursor, "GGUF", 4) != 0) {
        throw std::runtime_error("Invalid GGUF magic");
    }
    _cursor += 4;

    _version = read<uint32_t>();
    _num_tensors = read<int64_t>();
    _num_meta_kvs = read<int64_t>();
    _attributes_map = std::unordered_map<std::string, std::shared_ptr<GGUFKeyValue>>();
    _tensors_info_map = std::unordered_map<std::string, std::shared_ptr<GGUFTensorInfo>>();
}

void GGUFFileReader::readMetaKVs() {
    for (int64_t i = 0; i < _num_meta_kvs; ++i) {
        auto kv = std::make_shared<GGUFKeyValue>();
        kv->key = readString();
        kv->gguf_type = read<GGUF_TYPE>();

        if (kv->gguf_type == GGUF_TYPE_ARRAY) {
            GGUF_TYPE array_type = read<GGUF_TYPE>();
            uint64_t array_size = read<uint64_t>();
            kv->value.resize(array_size * GGUF_TYPE_SIZE[array_type]);
            kv->gguf_type = array_type;
            std::memcpy(kv->value.data(), _cursor, kv->value.size());
            _cursor += kv->value.size();
        } else if (kv->gguf_type == GGUF_TYPE_STRING) {
            uint64_t str_size = read<uint64_t>();
            kv->value.resize(str_size);
            std::memcpy(kv->value.data(), _cursor, str_size);
            _cursor += str_size;
        } else {
            kv->value.resize(GGUF_TYPE_SIZE[kv->gguf_type]);
            std::memcpy(kv->value.data(), _cursor, kv->value.size());
            _cursor += kv->value.size();
        }

        _meta_kvs.push_back(kv);
        _attributes_map.emplace(kv->key, kv);
    }
}

void GGUFFileReader::readTensorInfos() {
    for (int64_t i = 0; i < _num_tensors; ++i) {
        auto tensor_info = std::make_shared<GGUFTensorInfo>();
        tensor_info->name = readString();
        tensor_info->ndim = read<uint32_t>();
        tensor_info->shape.resize(tensor_info->ndim);
        for (size_t j = 0; j < tensor_info->ndim; ++j) {
            tensor_info->shape[j] = read<int64_t>();
        }
        tensor_info->ggml_type = read<GGML_TYPE>();
        tensor_info->data_offset = read<uint64_t>();
        _tensor_infos.push_back(tensor_info);
        _tensors_info_map.emplace(tensor_info->name, tensor_info);
    }
}

std::string GGUFFileReader::readString() {
    uint64_t length = read<uint64_t>();
    std::string str(reinterpret_cast<const char *>(_cursor), length);
    _cursor += length;
    return str;
}

template <typename T>
T GGUFFileReader::read() {
    T value;
    std::memcpy(&value, _cursor, sizeof(T));
    _cursor += sizeof(T);
    return value;
}

std::string GGUFFileReader::toString() const {
    std::ostringstream oss;
    oss << "GGUF File Contents: " << std::endl;
    oss << "Version: " << _version << std::endl;
    oss << "Number of Meta KVs: " << _num_meta_kvs << std::endl;
    oss << "Number of Tensors: " << _num_tensors << std::endl;
    if (isSplitFile()) {
        oss << "Split: " << getSplitNo() << " of " << getSplitCount() << std::endl;
    }
    oss << std::endl;
    oss << "Meta KVs: " << std::endl;
    for (const auto &kv : _meta_kvs) {
        oss << kv->toString() << std::endl;
    }
    oss << std::endl;
    oss << "Tensor INFOs: " << std::endl;
    for (const auto &info : _tensor_infos) {
        oss << info->toString() << std::endl;
    }
    return oss.str();
}

void *GGUFFileReader::getTensorData(const std::string &tensor_name) const {
    auto it = _tensors_info_map.find(tensor_name);
    if (it == _tensors_info_map.end()) {
        throw std::runtime_error("Tensor not found: " + tensor_name);
    }
    return getTensorData(*(it->second));
}

void *GGUFFileReader::getTensorData(const GGUFTensorInfo &tensor_info) const {
    if (!_data_start) {
        throw std::runtime_error("Data section not initialized");
    }
    return reinterpret_cast<uint8_t *>(_data_start) + tensor_info.data_offset;
}

size_t GGUFFileReader::getTensorSize(const std::string &tensor_name) const {
    auto it = _tensors_info_map.find(tensor_name);
    if (it == _tensors_info_map.end()) {
        throw std::runtime_error("Tensor not found: " + tensor_name);
    }
    return getTensorSize(*(it->second));
}

size_t GGUFFileReader::getTensorSize(const GGUFTensorInfo &tensor_info) const {
    return calculateTensorSize(tensor_info);
}

size_t GGUFFileReader::calculateTensorSize(const GGUFTensorInfo &tensor_info) const {
    // For quantized types, we need special handling
    // Based on GGML quantization block structure
    constexpr size_t QK_K = 256;

    size_t num_elements = 1;
    for (int64_t dim : tensor_info.shape) {
        num_elements *= dim;
    }

    switch (tensor_info.ggml_type) {
        case GGML_TYPE_F32:
            return num_elements * 4;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return num_elements * 2;
        case GGML_TYPE_F64:
            return num_elements * 8;
        case GGML_TYPE_I8:
            return num_elements * 1;
        case GGML_TYPE_I16:
            return num_elements * 2;
        case GGML_TYPE_I32:
            return num_elements * 4;
        case GGML_TYPE_I64:
            return num_elements * 8;
        // Quantized types - block-based
        case GGML_TYPE_Q4_0: {
            // Block size: 32, type size: 2 + 16 = 18 bytes per block
            size_t blocks = (num_elements + 31) / 32;
            return blocks * 18;
        }
        case GGML_TYPE_Q4_1: {
            // Block size: 32, type size: 2 + 2 + 16 = 20 bytes per block
            size_t blocks = (num_elements + 31) / 32;
            return blocks * 20;
        }
        case GGML_TYPE_Q5_0: {
            // Block size: 32, type size: 2 + 4 + 16 = 22 bytes per block
            size_t blocks = (num_elements + 31) / 32;
            return blocks * 22;
        }
        case GGML_TYPE_Q5_1: {
            // Block size: 32, type size: 2 + 2 + 4 + 16 = 24 bytes per block
            size_t blocks = (num_elements + 31) / 32;
            return blocks * 24;
        }
        case GGML_TYPE_Q8_0: {
            // Block size: 32, type size: 2 + 32 = 34 bytes per block
            size_t blocks = (num_elements + 31) / 32;
            return blocks * 34;
        }
        case GGML_TYPE_Q8_1: {
            // Block size: 32, type size: 4 + 4 + 32 = 40 bytes per block
            size_t blocks = (num_elements + 31) / 32;
            return blocks * 40;
        }
        case GGML_TYPE_Q2_K: {
            // Block size: 256, type size: 2 + 2 + QK_K/16 + QK_K/4 = 2 + 2 + 16 + 64 = 84 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 84;
        }
        case GGML_TYPE_Q3_K: {
            // Block size: 256, type size: 2 + QK_K/4 + QK_K/8 + 12 = 2 + 64 + 32 + 12 = 110 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 110;
        }
        case GGML_TYPE_Q4_K: {
            // Block size: 256, type size: 2 + 2 + QK_K/2 + 12 = 2 + 2 + 128 + 12 = 144 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 144;
        }
        case GGML_TYPE_Q5_K: {
            // Block size: 256, type size: 2 + 2 + QK_K/2 + QK_K/8 + 12 = 2 + 2 + 128 + 32 + 12 = 176 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 176;
        }
        case GGML_TYPE_Q6_K: {
            // Block size: 256, type size: 2 + QK_K/2 + QK_K/4 + QK_K/16 = 2 + 128 + 64 + 16 = 210 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 210;
        }
        case GGML_TYPE_Q8_K: {
            // Block size: 256, type size: 4 + QK_K + QK_K/8 = 4 + 256 + 32 = 292 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 292;
        }
        case GGML_TYPE_IQ2_XXS: {
            // Block size: 256, type size: 2 + QK_K/4 = 2 + 64 = 66 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 66;
        }
        case GGML_TYPE_IQ2_XS: {
            // Block size: 256, type size: 2 + QK_K/4 + QK_K/32 = 2 + 64 + 8 = 74 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 74;
        }
        case GGML_TYPE_IQ3_XXS: {
            // Block size: 256, type size: 2 + QK_K/4 + QK_K/8 = 2 + 64 + 32 = 98 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 98;
        }
        case GGML_TYPE_IQ1_S: {
            // Block size: 256, type size: 2 + QK_K/8 + QK_K/16 = 2 + 32 + 16 = 50 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 50;
        }
        case GGML_TYPE_IQ4_NL: {
            // Block size: 32, type size: 2 + 16 = 18 bytes per block
            size_t blocks = (num_elements + 31) / 32;
            return blocks * 18;
        }
        case GGML_TYPE_IQ3_S: {
            // Block size: 256, type size: 2 + QK_K/4 + QK_K/8 + QK_K/32 + 4 = 2 + 64 + 32 + 8 + 4 = 110 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 110;
        }
        case GGML_TYPE_IQ2_S: {
            // Block size: 256, type size: 2 + QK_K/4 + QK_K/16 = 2 + 64 + 16 = 82 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 82;
        }
        case GGML_TYPE_IQ4_XS: {
            // Block size: 256, type size: 2 + 2 + QK_K/2 + QK_K/64 = 2 + 2 + 128 + 4 = 136 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 136;
        }
        case GGML_TYPE_IQ1_M: {
            // Block size: 256, type size: QK_K/8 + QK_K/16 + QK_K/32 = 32 + 16 + 8 = 56 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 56;
        }
        case GGML_TYPE_TQ1_0: {
            // Block size: 256, type size: 2 + 4*13 = 2 + 52 = 54 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 54;
        }
        case GGML_TYPE_TQ2_0: {
            // Block size: 256, type size: 2 + 64 = 66 bytes per block
            size_t blocks = (num_elements + QK_K - 1) / QK_K;
            return blocks * 66;
        }
        default:
            throw std::runtime_error("Unsupported GGML_TYPE for size calculation: " +
                                    std::string(GGML_TYPE_NAME[tensor_info.ggml_type] ?
                                               GGML_TYPE_NAME[tensor_info.ggml_type] : "UNKNOWN"));
    }
}

size_t GGUFFileReader::getDataOffset() const {
    if (!_data_start) {
        return 0;
    }
    return reinterpret_cast<uintptr_t>(_data_start) - reinterpret_cast<uintptr_t>(_data);
}

bool GGUFFileReader::isSplitFile() const {
    return _attributes_map.find("split.count") != _attributes_map.end();
}

uint16_t GGUFFileReader::getSplitNo() const {
    auto it = _attributes_map.find("split.no");
    if (it == _attributes_map.end()) {
        return 0;
    }
    auto kv = it->second;
    if (kv->gguf_type == GGUF_TYPE_UINT16) {
        return *reinterpret_cast<const uint16_t *>(kv->value.data());
    }
    return 0;
}

uint16_t GGUFFileReader::getSplitCount() const {
    auto it = _attributes_map.find("split.count");
    if (it == _attributes_map.end()) {
        return 1;  // Not a split file, so count is 1
    }
    auto kv = it->second;
    if (kv->gguf_type == GGUF_TYPE_UINT16) {
        return *reinterpret_cast<const uint16_t *>(kv->value.data());
    }
    return 1;
}
