#include <catch2/catch_test_macros.hpp>

#include <core/cuda_check.hpp>
#include <core/tensor.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint16_t floatToBf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

float bf16BitsToFloat(std::uint16_t bf16) {
    std::uint32_t bits = static_cast<std::uint32_t>(bf16) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::vector<float> copyBf16ToFloat(const core::DeviceTensor& tensor) {
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

std::vector<float> copyF32ToFloat(const core::DeviceTensor& tensor) {
    size_t numel = 1;
    for (size_t dim : tensor.getShape()) {
        numel *= dim;
    }
    std::vector<float> host(numel);
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        tensor.getDevicePtr(),
        numel * sizeof(float),
        cudaMemcpyDeviceToHost));
    return host;
}

std::vector<std::uint8_t> copyRawBytes(const core::DeviceTensor& tensor, size_t nbytes) {
    std::vector<std::uint8_t> host(nbytes);
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        tensor.getDevicePtr(),
        nbytes,
        cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace

TEST_CASE("host upload BF16 roundtrip", "[tensor][host]") {
    const std::vector<size_t> shape{2, 3};
    const std::vector<float> values{1.f, 2.f, 0.5f, -1.f, 0.f, 4.f};
    std::vector<std::uint16_t> host(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        host[i] = floatToBf16(values[i]);
    }

    core::DeviceTensor t(host.data(), shape, "bf16_host", core::DType::BF16);

    REQUIRE(t.getShape() == shape);
    REQUIRE(t.getName() == "bf16_host");
    REQUIRE(t.getDtype() == core::DType::BF16);
    REQUIRE(t.getDevicePtr() != nullptr);

    const std::vector<float> got = copyBf16ToFloat(t);
    REQUIRE(got.size() == values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        REQUIRE(got[i] == bf16BitsToFloat(floatToBf16(values[i])));
    }
}

TEST_CASE("host upload F32 roundtrip", "[tensor][host]") {
    const std::vector<size_t> shape{2, 2};
    const std::vector<float> values{1.25f, -2.5f, 0.f, 8.f};

    core::DeviceTensor t(values.data(), shape, "f32_host", core::DType::F32);

    REQUIRE(t.getShape() == shape);
    REQUIRE(t.getName() == "f32_host");
    REQUIRE(t.getDtype() == core::DType::F32);
    REQUIRE(t.getDevicePtr() != nullptr);
    REQUIRE(copyF32ToFloat(t) == values);
}

TEST_CASE("host upload rejects null pointer", "[tensor][host]") {
    REQUIRE_THROWS_AS(
        core::DeviceTensor(nullptr, {2, 2}, "null", core::DType::F32),
        std::invalid_argument);
}

TEST_CASE("empty allocates writable device buffer", "[tensor][factory]") {
    auto t = core::DeviceTensor::empty({2, 2}, "empty_bf16", core::DType::BF16);

    REQUIRE(t.getShape() == std::vector<size_t>({2, 2}));
    REQUIRE(t.getName() == "empty_bf16");
    REQUIRE(t.getDtype() == core::DType::BF16);
    REQUIRE(t.getDevicePtr() != nullptr);

    const std::uint16_t pattern = floatToBf16(3.f);
    CUDA_CHECK(cudaMemcpy(
        t.getDevicePtr(),
        &pattern,
        sizeof(pattern),
        cudaMemcpyHostToDevice));

    std::uint16_t got = 0;
    CUDA_CHECK(cudaMemcpy(
        &got,
        t.getDevicePtr(),
        sizeof(got),
        cudaMemcpyDeviceToHost));
    REQUIRE(got == pattern);
}

TEST_CASE("zeroes BF16 and F32 are all zero bytes", "[tensor][factory]") {
    SECTION("BF16 shape {4} -> 8 zero bytes") {
        auto t = core::DeviceTensor::zeroes({4}, "z_bf16", core::DType::BF16);
        REQUIRE(t.getDevicePtr() != nullptr);
        const auto bytes = copyRawBytes(t, 8);
        REQUIRE(bytes.size() == 8);
        for (std::uint8_t b : bytes) {
            REQUIRE(b == 0);
        }
        for (float v : copyBf16ToFloat(t)) {
            REQUIRE(v == 0.f);
        }
    }

    SECTION("F32 shape {4} -> 16 zero bytes") {
        auto t = core::DeviceTensor::zeroes({4}, "z_f32", core::DType::F32);
        REQUIRE(t.getDevicePtr() != nullptr);
        const auto bytes = copyRawBytes(t, 16);
        REQUIRE(bytes.size() == 16);
        for (std::uint8_t b : bytes) {
            REQUIRE(b == 0);
        }
        for (float v : copyF32ToFloat(t)) {
            REQUIRE(v == 0.f);
        }
    }
}

TEST_CASE("move construct transfers usable ownership", "[tensor][move]") {
    core::DeviceTensor src = core::DeviceTensor::zeroes({3}, "move_src", core::DType::F32);
    const void* srcPtr = src.getDevicePtr();
    core::DeviceTensor dst(std::move(src));

    REQUIRE(dst.getDevicePtr() == srcPtr);
    REQUIRE(dst.getShape() == std::vector<size_t>({3}));
    REQUIRE(dst.getName() == "move_src");
    REQUIRE(dst.getDtype() == core::DType::F32);
    for (float v : copyF32ToFloat(dst)) {
        REQUIRE(v == 0.f);
    }
    // src and dst destroy cleanly at end of scope (moved-from ptr is null).
}

TEST_CASE("move assign transfers usable ownership", "[tensor][move]") {
    core::DeviceTensor a = core::DeviceTensor::zeroes({2}, "a", core::DType::F32);
    core::DeviceTensor b = core::DeviceTensor::empty({2}, "b", core::DType::F32);

    const float fill = 7.f;
    CUDA_CHECK(cudaMemcpy(b.getDevicePtr(), &fill, sizeof(fill), cudaMemcpyHostToDevice));
    // second element
    CUDA_CHECK(cudaMemcpy(
        static_cast<std::uint8_t*>(b.getDevicePtr()) + sizeof(float),
        &fill,
        sizeof(fill),
        cudaMemcpyHostToDevice));

    a = std::move(b);

    REQUIRE(a.getName() == "b");
    REQUIRE(a.getShape() == std::vector<size_t>({2}));
    const auto vals = copyF32ToFloat(a);
    REQUIRE(vals.size() == 2);
    REQUIRE(vals[0] == 7.f);
    REQUIRE(vals[1] == 7.f);
}

TEST_CASE("const getDevicePtr supports D2H", "[tensor][host]") {
    const std::vector<float> values{1.f, 2.f};
    core::DeviceTensor t(values.data(), {2}, "const_ptr", core::DType::F32);
    const core::DeviceTensor& cref = t;

    std::vector<float> host(2);
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        cref.getDevicePtr(),
        2 * sizeof(float),
        cudaMemcpyDeviceToHost));
    REQUIRE(host == values);
}
