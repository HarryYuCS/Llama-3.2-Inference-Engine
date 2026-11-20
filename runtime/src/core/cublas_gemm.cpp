#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <core/tensor.hpp>
#include <core/cublas_gemm.hpp>

#include <stdexcept>

namespace core {

CublasGemmExecutor::CublasGemmExecutor() {
    // initialize cublas handle
    cublasStatus_t status;

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Cublas handle creation failed.");
    }
}

void CublasGemmExecutor::linear(
    const core::DeviceTensor &X,
    const core::DeviceTensor &W,
    DeviceTensor &Y
) {
    // fail if tensors don't match dtype
    core::DType dtypeX = X.getDtype();
    core::DType dtypeW = W.getDtype();
    if (dtypeX != dtypeW) {
        throw std::invalid_argument("Tensor dtypes do not match: " + X.getName() + " " + W.getName());
    }
    core::DType gemmDtype = dtypeX;

    // cublas transpositions (reading memory as column major):
    // X -> X^T has dimensions [K, M], leading dim K
    // W -> W^T B has dimensions [K, N], leading dim K
    // Y -> Y^T C has dimensions [N, M], leading dim N
    // We technincally want Y = X @ W^T, but the memory storage is the barrier
    // Thus, we calculate Y^T = W @ X^T, so we undo transpose the cublas read for W, keep it for X
    // Then, writing Y^T to Y in column major format is the row major memory mapping for Y.
    std::vector<size_t> xShape = X.getShape();
    std::vector<size_t> wShape = W.getShape();
    std::vector<size_t> yShape = Y.getShape();
    size_t M = xShape[0];
    size_t K = xShape[1];
    size_t N = wShape[0];
    // fail if shapes don't match TODO
    if (wShape[1] != K) {
        throw std::invalid_argument("K dimension doesn't match");
    }
    if (yShape[0] != M) {
        throw std::invalid_argument("Output M dimension doesn't match");
    }
    if (yShape[1] != N) {
        throw std::invalid_argument("Output N dimensions doesn't match");
    }

    // device pointers
    const void* xDevicePtr = X.getDevicePtr();
    const void* wDevicePtr = W.getDevicePtr();
    void* yDevicePtr = Y.getDevicePtr();

    float alpha = 1.0f; // cublas : alpha(AB) + beta(C)
    float beta = 0.0f;

    // only support BF16, leave the scaffolding for completeness
    if (gemmDtype == core::DType::BF16) {
        cublasStatus_t status = cublasGemmEx(
            handle,
            CUBLAS_OP_T, // W view is W^T; OP_T restores math W
            CUBLAS_OP_N, // X view is already X^T
            N, // N, M switched since everything transposed and switched
            M,
            K,
            &alpha,
            wDevicePtr, CUDA_R_16BF, K, // leading dim for B is N
            xDevicePtr, CUDA_R_16BF, K, // leading dim for A is K
            &beta,
            yDevicePtr, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Cublas gemm failed.");
        }
    }
    else {
        throw std::invalid_argument("Only BF16 is supported at the moment");
    }
}

// clean cublas handle in destructor
CublasGemmExecutor::~CublasGemmExecutor() {
    cublasDestroy(handle);
}

} // namespace core