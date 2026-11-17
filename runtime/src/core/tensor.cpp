#include <core/tensor.hpp>
#include <core/cuda_check.hpp>

#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>

namespace core {

DeviceTensor::DeviceTensor(
    void* hostPtr,
    const std::vector<size_t> &shape,
    const std::string &name,
    DType dtype
) :  
    shape_(shape),
    name_(name),
    dtype_(dtype)
{
    // calc total copy size
    int elementBytes{0};
    switch (dtype_) {
        case BF16:
        case F16:
            elementBytes = 2;
            break;
        case F32:
            elementBytes = 4;
            break;
    }

    // calc total elements
    int totalElements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>);

    int totalSize = elementSize * totalElements;

    CUDA_CHECK(cudaMalloc(&devicePtr_, totalSize));
    CUDA_CHECK(cudaMemcpy(devicePtr_, hostPtr, totalSize, cudaMemcpyHostToDevice));
}

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept {
    devicePtr_ = other.devicePtr_;
    other.devicePtr_ = nullptr;

    shape_ = std::move(other.shape_);
    name_ = std::move(other.name_);
    dtype_ = other.dtype_;
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
    // check reflexivity
    if (this == &other) return *this;

    // free own if available
    if (devicePtr_) {
        CUDA_CHECK(cudaFree(devicePtr_));
    }

    devicePtr_ = other.devicePtr_;
    other.devicePtr_ = nullptr;

    shape_ = std::move(other.shape_);
    name_ = std::move(other.name_);
    dtype_ = other.dtype_;
    
    return *this;
}

const void *getDevicePtr() {
    return devicePtr_;
}
const std::vector<size_t> &getShape() {
    return shape_;
}
const std::string &getName() {
    return name_;
}
DType getDtype() {
    return dtype_;
}

DeviceTensor::~DeviceTensor() {
    if (devicePtr_) {
        CUDA_CHECK(cudaFree(devicePtr_));
    }
}

}