#include <model/weight_store.hpp>
#include <model/model_config.hpp>

#include <core/cuda_check.hpp>
#include <core/tensor.hpp>

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <iostream>
#include <unordered_map>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>

/**

parse safetensors, initialize tensors for each layer

bytes 0-7: header size (N) of the JSON header
bytes 8 to 8 + N - 1: JSON header UTF-8 containing tensor names, data types, data offsets
bytes 8 + N to EOF: raw data buffers of the tensor weights

*/

namespace model {

namespace {

// utility to convert string to dtype enum
core::DType strToDtype(const std::string &s) {
    if (s == "BF16") {
        return core::DType::BF16;
    }
    else if (s == "F16") {
        return core::DType::F16;
    }
    else if (s == "F32") {
        return core::DType::F32;
    }
    else {
        throw std::invalid_argument("Must be either BF16, F16, or F32.");
    }
}

} // anonymous namespace

WeightStore::WeightStore(const std::string &safetensorsPath) {
    int fd = open(safetensorsPath.data(), O_RDONLY);
    struct stat st{};
    fstat(fd, &st);
    size_t fileSize = st.st_size;

    void* mapped = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);

    close(fd);

    auto* fileBegin = static_cast<const uint8_t*>(mapped);

    uint64_t headerLength{0};
    std::memcpy(&headerLength, fileBegin, 8); // dest, src, count

    // read as char for nlohmann
    const char* jsonBegin = reinterpret_cast<const char*>(fileBegin + 8);
    // debug print
    nlohmann::json header = nlohmann::json::parse(std::string_view(jsonBegin, headerLength));

    const uint8_t* tensorBegin = fileBegin + 8 + headerLength;

    for (auto& [name, data] : header.items()) {
        if (name == "__metadata__") continue;

        std::string dtype = data.at("dtype").get<std::string>();
        std::vector<size_t> shape = data.at("shape").get<std::vector<size_t>>();
        std::vector<size_t> offsets = data.at("data_offsets").get<std::vector<size_t>>();
        size_t start = offsets[0], end = offsets[1]; // offsets is a pair of start, end

        const void* hostPtr = tensorBegin + start;
        size_t tensorNumBytes = end - start;

        std::cout << "[DEBUG] Tensor " << name << ", dtype " << dtype << " at offset " << start << std::endl;
        tensors_.insert({
            name,
            core::DeviceTensor(
                hostPtr,
                shape,
                name,
                strToDtype(dtype)
            )
        });
    }

    munmap(mapped, fileSize);
}

const core::DeviceTensor &WeightStore::get(const std::string &name) const {
    if (!tensors_.contains(name)) {
        throw std::invalid_argument("No tensor of that name exists in the weight store.");
    }

    return tensors_.at(name);
}

}