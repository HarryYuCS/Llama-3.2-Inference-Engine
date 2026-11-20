#include <catch2/catch_test_macros.hpp>

#include <core/cublas_gemm.hpp>
#include <core/cuda_check.hpp>
#include <core/tensor.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint16_t floatToBf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round-to-nearest-even via +0x7fff with sticky bit from lower 16.
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

float bf16BitsToFloat(std::uint16_t bf16) {
    std::uint32_t bits = static_cast<std::uint32_t>(bf16) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

core::DeviceTensor makeBf16Tensor(
    const std::vector<float>& values,
    const std::vector<size_t>& shape,
    const std::string& name) {
    size_t numel = 1;
    for (size_t dim : shape) {
        numel *= dim;
    }
    REQUIRE(values.size() == numel);

    std::vector<std::uint16_t> host(numel);
    for (size_t i = 0; i < numel; ++i) {
        host[i] = floatToBf16(values[i]);
    }

    return core::DeviceTensor(host.data(), shape, name, core::DType::BF16);
}

std::vector<float> copyBf16TensorToFloat(const core::DeviceTensor& tensor) {
    size_t numel = 1;
    for (size_t dim : tensor.getShape()) {
        numel *= dim;
    }

    std::vector<std::uint16_t> host(numel);
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        tensor.getDevicePtr(),
        numel * sizeof(std::uint16_t),
        cudaMemcpyDeviceToHost));

    std::vector<float> out(numel);
    for (size_t i = 0; i < numel; ++i) {
        out[i] = bf16BitsToFloat(host[i]);
    }
    return out;
}

void requireClose(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1e-2f) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(std::fabs(actual[i] - expected[i]) <= atol);
    }
}

}  // namespace

TEST_CASE("linear: Y = X @ W.T for square identity activations", "[cublas][linear]") {
    // X [2,2] = I, W [2,2] = [[1,2],[3,4]] (rows = out)
    // Y = X @ W.T = W.T = [[1,3],[2,4]]
    auto X = makeBf16Tensor({1.f, 0.f, 0.f, 1.f}, {2, 2}, "X");
    auto W = makeBf16Tensor({1.f, 2.f, 3.f, 4.f}, {2, 2}, "W");
    auto Y = makeBf16Tensor({0.f, 0.f, 0.f, 0.f}, {2, 2}, "Y");

    core::CublasGemmExecutor gemm;
    REQUIRE_NOTHROW(gemm.linear(X, W, Y));

    requireClose(copyBf16TensorToFloat(Y), {1.f, 3.f, 2.f, 4.f});
}

TEST_CASE("linear: Y = X @ W.T for decode-shaped [1,K] x [N,K]", "[cublas][linear]") {
    // X [1,3] = [1,2,3]
    // W [2,3] = [[1,0,0],[0,1,0]]
    // Y = X @ W.T = [1, 2]
    auto X = makeBf16Tensor({1.f, 2.f, 3.f}, {1, 3}, "X");
    auto W = makeBf16Tensor({1.f, 0.f, 0.f, 0.f, 1.f, 0.f}, {2, 3}, "W");
    auto Y = makeBf16Tensor({0.f, 0.f}, {1, 2}, "Y");

    core::CublasGemmExecutor gemm;
    REQUIRE_NOTHROW(gemm.linear(X, W, Y));

    requireClose(copyBf16TensorToFloat(Y), {1.f, 2.f});
}

TEST_CASE("linear: rejects mismatched K", "[cublas][linear]") {
    auto X = makeBf16Tensor({1.f, 2.f}, {1, 2}, "X");
    auto W = makeBf16Tensor({1.f, 2.f, 3.f}, {1, 3}, "W");  // K=3 != 2
    auto Y = makeBf16Tensor({0.f}, {1, 1}, "Y");

    core::CublasGemmExecutor gemm;
    REQUIRE_THROWS_AS(gemm.linear(X, W, Y), std::invalid_argument);
}
