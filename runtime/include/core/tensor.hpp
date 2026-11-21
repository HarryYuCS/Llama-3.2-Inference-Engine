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

    // torch-like factories
    static DeviceTensor empty(
        const std::vector<size_t> &shape,
        const std::string &name,
        DType dtype
    );
    static DeviceTensor zeroes(
        const std::vector<size_t> &shape,
        const std::string &name,
        DType dtype
    );

    DeviceTensor(const DeviceTensor&) = delete; // no copy assignment for now (double free)
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    // moves to be implemented
    DeviceTensor(DeviceTensor&& other) noexcept;
    DeviceTensor& operator=(DeviceTensor&&) noexcept;

    const void *getDevicePtr() const;
    void *getDevicePtr(); // non const overload

    const std::vector<size_t> &getShape() const;
    const std::string &getName() const;
    DType getDtype() const;

    ~DeviceTensor(); // free the device ptr
private:
    // private constructor with no memory allocation - hidden to avoid memory issues
    DeviceTensor(
        const std::vector<size_t> &shape,
        const std::string &name,
        DType dtype
    );

    // get Byte size
    size_t getNumBytes() const;

    void* devicePtr_ = nullptr;
    std::vector<size_t> shape_;

    std::string name_;
    DType dtype_;
};

} // namespace core

#endif // CORE_TENSOR_HPP