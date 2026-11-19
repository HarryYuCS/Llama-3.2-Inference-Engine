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

    // matches tensor dtypes, then writes to an output tensor
    void gemm(
        const core::DeviceTensor &A,
        const core::DeviceTensor &B,
        DeviceTensor &C,
        bool transposeB = true
    );
private:
    cublasHandle_t handle;
};

} // namespace core

#endif // CUBLAS_GEMM_HPP

