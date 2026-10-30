cmake_minimum_required(VERSION 3.14)
project(NeuralNetwork CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# Libreria con la clase Matrix
add_library(matrix matrix.cpp)
target_include_directories(matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Experimento de la neurona (Fase 2)
add_executable(neuron neuron.cpp)

# Tests (GoogleTest)
enable_testing()
find_package(GTest REQUIRED)
add_executable(test_matrix tests/test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE matrix GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_matrix)
