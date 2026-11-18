/**

Tensor abstraction wrapping a pointer to device

*/

#ifndef CORE_TENSOR_HPP
#define CORE_TENSOR_HPP

#include <cuda_runtime.h>
#include <core/cuda_check.hpp>

#include <vector>
#include <string>

namespace core {

// for completeness of tensor class, even though weights are only BF16
enum class DType {
    BF16,
    F16,
    F32
};

class DeviceTensor {
public:
    DeviceTensor(
        const void* hostPtr,
        const std::vector<size_t> &shape,
        const std::string &name,
        DType dtype
    ); // constructs tensor using data from host, makes a device copy and owns it

    DeviceTensor(const DeviceTensor&) = delete; // no copy assignment for now (double free)
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    // moves to be implemented
    DeviceTensor(DeviceTensor&& other) noexcept;
    DeviceTensor& operator=(DeviceTensor&&) noexcept;

    const void *getDevicePtr();
    const std::vector<size_t> &getShape();
    const std::string &getName();
    DType getDtype();

    ~DeviceTensor(); // free the device ptr
private:
    void* devicePtr_ = nullptr;
    std::vector<size_t> shape_;

    std::string name_;
    DType dtype_;
};

} // namespace core

#endif // CORE_TENSOR_HPP