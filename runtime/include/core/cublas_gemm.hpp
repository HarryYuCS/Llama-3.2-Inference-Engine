/**

wrapper for cuBLAS GEMM that transposes loaded weights (safetensors is )

*/

#ifndef CUBLAS_GEMM_HPP
#define CUBLAS_GEMM_HPP

#include <cublas_v2.h>

#include <core/tensor.hpp>
#include <core/cuda_check.hpp>

namespace core {

class CublasGemmExecutor {
public:
    // initialize cublas handle
    CublasGemmExecutor();

    // no copy or move operations
    CublasGemmExecutor(const CublasGemmExecutor &) = delete;
    CublasGemmExecutor &operator=(const CublasGemmExecutor &) = delete;
    CublasGemmExecutor(CublasGemmExecutor &&) noexcept = delete;
    CublasGemmExecutor &operator=(CublasGemmExecutor &&) noexcept = delete;

    ~CublasGemmExecutor();

    /**
    * @brief Computes row-major memory layout gemm with weights (B) as [output, input] and writes row-major to C
    *   
    * @param A : tensor with row major memory layout
    * @param B : tensor with row major memory layout
    */
    void linear(
        const core::DeviceTensor &X,
        const core::DeviceTensor &W,
        DeviceTensor &Y
    );

private:
    cublasHandle_t handle = nullptr;
};

} // namespace core

#endif // CUBLAS_GEMM_HPP

