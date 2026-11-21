#include <core/tensor.hpp>
#include <core/cuda_check.hpp>

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>

namespace core {

DeviceTensor::DeviceTensor(
    const void* hostPtr,
    const std::vector<size_t> &shape,
    const std::string &name,
    DType dtype
) :  
    shape_(shape),
    name_(name),
    dtype_(dtype)
{
    if (!hostPtr) {
        throw std::invalid_argument("Host pointer cannot be null.");
    }

    const size_t totalSize = getNumBytes();

    CUDA_CHECK(cudaMalloc(&devicePtr_, totalSize));
    CUDA_CHECK(cudaMemcpy(devicePtr_, hostPtr, totalSize, cudaMemcpyHostToDevice));
}

// private constructor
DeviceTensor::DeviceTensor(
    const std::vector<size_t> &shape,
    const std::string &name,
    DType dtype
) :
    shape_(shape),
    name_(name),
    dtype_(dtype)
{}

DeviceTensor DeviceTensor::empty(
    const std::vector<size_t> &shape,
    const std::string &name,
    DType dtype
) {
    DeviceTensor t(shape, name, dtype);
    CUDA_CHECK(cudaMalloc(&t.devicePtr_, t.getNumBytes()));
    return t;
}

DeviceTensor DeviceTensor::zeroes(
    const std::vector<size_t> &shape,
    const std::string &name,
    DType dtype
) {
    DeviceTensor t = empty(shape, name, dtype);
    CUDA_CHECK(cudaMemset(t.devicePtr_, 0, t.getNumBytes()));
    return t;
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
        cudaFree(devicePtr_);
    }

    devicePtr_ = other.devicePtr_;
    other.devicePtr_ = nullptr;

    shape_ = std::move(other.shape_);
    name_ = std::move(other.name_);
    dtype_ = other.dtype_;
    
    return *this;
}

const void *DeviceTensor::getDevicePtr() const {
    return devicePtr_;
}
void *DeviceTensor::getDevicePtr() {
    return devicePtr_;
}

const std::vector<size_t> &DeviceTensor::getShape() const {
    return shape_;
}
const std::string &DeviceTensor::getName() const {
    return name_;
}
DType DeviceTensor::getDtype() const {
    return dtype_;
}

size_t DeviceTensor::getNumBytes() const {
    // calc total copy size
    size_t elementBytes{0};
    switch (dtype_) {
        case DType::BF16:
        case DType::F16:
            elementBytes = 2;
            break;
        case DType::F32:
            elementBytes = 4;
            break;
    }

    // calc total elements
    size_t totalElements = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>{});

    size_t totalSize = elementBytes * totalElements;

    return totalSize;
}

DeviceTensor::~DeviceTensor() {
    if (devicePtr_) {
        cudaFree(devicePtr_);
    }
}

}