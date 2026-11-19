#include <catch2/catch_test_macros.hpp>

#include <model/weight_store.hpp>
#include <core/tensor.hpp>
#include <core/cuda_check.hpp>

#include <concepts>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifndef TEST_WEIGHTS_DIR
#error "TEST_WEIGHTS_DIR must be set by CMake"
#endif

// c-style to std::string
const std::string weightsDir{TEST_WEIGHTS_DIR};
// one weights store object
model::WeightStore weightStore(weightsDir + "/model.safetensors");

// comparison utility (hybrid epsilon)
template <std::floating_point T>
constexpr bool nearlyEqual(
    T a,
    T b,
    T absEpsilon = T{1e-12},
    T relEpsilon = std::numeric_limits<T>::epsilon() * T{8}) {
    if (a == b) return true;
    if (std::isnan(a) || std::isnan(b)) return false;

    const T diff = std::abs(a - b);
    if (diff <= absEpsilon) {
        return true;
    }

    const T largest = std::max(std::abs(a), std::abs(b));
    return diff <= largest * relEpsilon;
}

namespace {

float firstBf16AsFloat(const core::DeviceTensor& tensor) {
    REQUIRE(tensor.getDtype() == core::DType::BF16);

    std::uint16_t bf16Bits = 0;
    CUDA_CHECK(cudaMemcpy(
        &bf16Bits,
        tensor.getDevicePtr(),
        sizeof(bf16Bits),
        cudaMemcpyDeviceToHost));

    // BF16 is the upper 16 bits of IEEE-754 float32.
    std::uint32_t f32Bits = static_cast<std::uint32_t>(bf16Bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &f32Bits, sizeof(value));
    return value;
}

void requireTensor(
    const std::string& name,
    const std::vector<size_t>& expectedShape,
    float expectedFirst) {
    REQUIRE_NOTHROW(weightStore.get(name));

    const core::DeviceTensor& tensor = weightStore.get(name);
    const std::vector<size_t>& shape = tensor.getShape();

    REQUIRE(shape == expectedShape);
    REQUIRE(nearlyEqual(firstBf16AsFloat(tensor), expectedFirst));
}

}  // namespace

TEST_CASE("Ensure embed_tokens layer exists, and has the right shape", "[weightStore]") {
    REQUIRE_NOTHROW(weightStore.get("model.embed_tokens.weight"));

    const core::DeviceTensor& embed = weightStore.get("model.embed_tokens.weight");
    const std::vector<size_t> shape = embed.getShape();

    REQUIRE(shape.size() == 2);
    REQUIRE(shape[0] == 128256);
    REQUIRE(shape[1] == 2048);
}

TEST_CASE("Ensure transformer block 0 has correct shapes and first elements", "[weightStore]") {
    // Values from safetensors / check_safetensors.py (BF16 → float).
    // input RMS norm
    requireTensor("model.layers.0.input_layernorm.weight", {2048}, 0.154296875f);

    // GQA attention
    requireTensor("model.layers.0.self_attn.q_proj.weight", {2048, 2048}, -0.0179443359375f);
    requireTensor("model.layers.0.self_attn.k_proj.weight", {512, 2048}, 0.05810546875f);
    requireTensor("model.layers.0.self_attn.v_proj.weight", {512, 2048}, 0.014892578125f);
    requireTensor("model.layers.0.self_attn.o_proj.weight", {2048, 2048}, 0.00010919570922851562f);
    requireTensor("model.layers.0.post_attention_layernorm.weight", {2048}, 0.197265625f);

    // MLP / SwiGLU
    requireTensor("model.layers.0.mlp.up_proj.weight", {8192, 2048}, -0.0262451171875f);
    requireTensor("model.layers.0.mlp.gate_proj.weight", {8192, 2048}, 0.0260009765625f);
    requireTensor("model.layers.0.mlp.down_proj.weight", {2048, 8192}, -0.02685546875f);
}
