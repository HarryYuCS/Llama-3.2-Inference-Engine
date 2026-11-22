/**

Simple Linear Layer

*/

#ifndef LINEAR_HPP
#define LINEAR_HPP

#include <core/tensor.hpp>
#include <core/cublas_gemm.hpp>

namespace model {

class Linear {
public:
    Linear(
        core::CublasGemmExecutor& gemmExecutor,
        const core::DeviceTensor& weights
    );

    Linear(const Linear&) = delete;
    Linear& operator=(const Linear&) = delete;
    Linear(Linear&&) noexcept = delete;
    Linear& operator=(Linear&&) noexcept = delete;

    void forward(
        const core::DeviceTensor& input,
        core::DeviceTensor& output
    ) const;

    ~Linear() = default;

private:
    core::CublasGemmExecutor& gemmExecutor_;
    const core::DeviceTensor& weights_;
};

}  // namespace model

#endif  // LINEAR_HPP
