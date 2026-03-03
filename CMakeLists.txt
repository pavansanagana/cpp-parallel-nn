cmake_minimum_required(VERSION 3.20)
project(cpp_parallel_nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer LLVM clang on macOS for OpenMP
# We'll pass it explicitly when configuring (see build commands below).

find_package(OpenMP REQUIRED)

add_executable(bench_matmul
  src/bench_matmul.cpp
  src/matmul.cpp
)

target_include_directories(bench_matmul PRIVATE include)
target_link_libraries(bench_matmul PRIVATE OpenMP::OpenMP_CXX)

# Strong perf flags (safe defaults)
target_compile_options(bench_matmul PRIVATE -O3 -march=native)
add_executable(train_mlp
  src/train_mlp.cpp
  src/nn.cpp
  src/matmul.cpp
)

target_include_directories(train_mlp PRIVATE include)
target_link_libraries(train_mlp PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(train_mlp PRIVATE -O3 -march=native)
