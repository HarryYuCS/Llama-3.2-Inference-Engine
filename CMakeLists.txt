cmake_minimum_required(VERSION 3.28)
project(LlamaEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# external packages
find_package(nlohmann_json 3.11.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Catch2 3 REQUIRED)

# build kernels separately as a library
# TODO

# build engine, model as a single library for tests
add_library(llama_engine_lib STATIC
    # core
    ${PROJECT_SOURCE_DIR}/runtime/src/core/tensor.cpp
    # model
    ${PROJECT_SOURCE_DIR}/runtime/src/model/model_config.cpp
    ${PROJECT_SOURCE_DIR}/runtime/src/model/weight_store.cpp
    # engine
)
target_include_directories(llama_engine_lib
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/runtime/include
)
target_link_libraries(llama_engine_lib
    PUBLIC
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cublas
)

# build the main binary
add_executable(llama_engine
    ${PROJECT_SOURCE_DIR}/runtime/src/main.cpp
)
target_link_libraries(llama_engine
    PRIVATE
        llama_engine_lib
)

# build tests
enable_testing()
add_executable(llama_engine_tests
    # model
    ${PROJECT_SOURCE_DIR}/runtime/tests/model/test_weight_store.cpp
    ${PROJECT_SOURCE_DIR}/runtime/tests/model/test_model_config.cpp
)
target_link_libraries(llama_engine_tests
    PRIVATE 
        llama_engine_lib
        Catch2::Catch2WithMain
)
# bake in the weights dir
target_compile_definitions(llama_engine_tests
    PRIVATE
        TEST_WEIGHTS_DIR="${PROJECT_SOURCE_DIR}/weights"
)
include(Catch)
catch_discover_tests(llama_engine_tests)




