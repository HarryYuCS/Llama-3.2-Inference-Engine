#include <catch2/catch_test_macros.hpp>
#include <model/model_config.hpp>

#include <string>

#ifndef TEST_WEIGHTS_DIR
#error "TEST_WEIGHTS_DIR must be set by CMake"
#endif

// c-style to std:string 
const std::string weightsDir{TEST_WEIGHTS_DIR};
// one model config object
model::ModelConfig cfg(weightsDir + "/config.json");

TEST_CASE ("Test config hidden size.", "[config]") {
    REQUIRE(cfg.hiddenSize == 2048);
}