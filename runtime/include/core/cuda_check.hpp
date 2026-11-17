#ifndef CUDA_CHECK_HPP
#define CUDA_CHECK_HPP

#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr_to_check)                                          \
    do {                                                                   \
        cudaError_t result = (expr_to_check);                              \
        if (result != cudaSuccess) {                                       \
            const char* msg = cudaGetErrorString(result);                  \
            fprintf(stderr,                                                \
                    "CUDA Runtime Error: %s:%d: %s (%d)\n",                \
                    __FILE__,                                              \
                    __LINE__,                                              \
                    msg,                                                   \
                    static_cast<int>(result));                             \
            throw std::runtime_error(std::string("CUDA error: ") + msg);   \
        }                                                                  \
    } while (0)

#endif  // CUDA_CHECK_HPP
