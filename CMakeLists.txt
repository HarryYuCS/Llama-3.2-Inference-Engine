cmake_minimum_required(VERSION 3.28)
project(LlamaEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# external packages
find_package(nlohmann_json 3.11.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# build kernels separately as a library
# TODO

# build engine, model as a single library for tests
add_library(LlamaEngineLib STATIC
    # core
    ${PROJECT_SOURCE_DIR}/runtime/src/core/tensor.cpp
    # model
    ${PROJECT_SOURCE_DIR}/runtime/src/model/model_config.cpp
    ${PROJECT_SOURCE_DIR}/runtime/src/model/weight_store.cpp
    # engine
)
target_include_directories(LlamaEngineLib
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/runtime/include
)
target_link_libraries(LlamaEngineLib
    PUBLIC
        nlohmann_json::nlohmann_json
        CUDA::cudart
)

# build the main binary
add_executable(LlamaEngine
    ${PROJECT_SOURCE_DIR}/runtime/src/main.cpp
)
target_link_libraries(LlamaEngine
    PRIVATE
        LlamaEngineLib
)

# build tests





