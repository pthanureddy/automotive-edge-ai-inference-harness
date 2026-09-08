cmake_minimum_required(VERSION 3.24)
project(automotive_edge_ai_inference_harness LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(edge_ai_model STATIC firmware/src/model.c)
target_include_directories(
    edge_ai_model
    PUBLIC firmware/include
    PRIVATE firmware/generated
)
if(MSVC)
    target_compile_options(edge_ai_model PRIVATE /W4 /WX)
else()
    target_compile_options(edge_ai_model PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(edge_ai_model PUBLIC m)
endif()

add_executable(edge_ai_tests firmware/tests/test_inference.cpp)
target_link_libraries(edge_ai_tests PRIVATE edge_ai_model)
if(MSVC)
    target_compile_options(edge_ai_tests PRIVATE /W4 /WX)
else()
    target_compile_options(edge_ai_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(edge_ai_benchmark firmware/app/benchmark.cpp)
target_link_libraries(edge_ai_benchmark PRIVATE edge_ai_model)
if(MSVC)
    target_compile_options(edge_ai_benchmark PRIVATE /W4 /WX)
else()
    target_compile_options(edge_ai_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

enable_testing()
add_test(
    NAME reference_vectors
    COMMAND edge_ai_tests ${CMAKE_SOURCE_DIR}/artifacts/reference_vectors.csv
)
