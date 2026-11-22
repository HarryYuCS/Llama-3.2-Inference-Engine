#include <model/linear.hpp>

namespace model {

Linear::Linear(
    core::CublasGemmExecutor& gemmExecutor,
    const core::DeviceTensor& weights
) :
    gemmExecutor_(gemmExecutor),
    weights_(weights)
{}

void Linear::forward(
    const core::DeviceTensor& input,
    core::DeviceTensor& output
) const {
    // CublasGemmExecutor validates the GEMM.
    gemmExecutor_.linear(input, weights_, output);
}

}  // namespace model
