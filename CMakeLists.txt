cmake_minimum_required(VERSION 3.16)
project(cpp_parallel_nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bench src/bench.cpp)
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bench PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(bench PRIVATE HAS_OPENMP=1)
endif()

# Optional CUDA (build this on a machine with NVIDIA + nvcc)
option(USE_CUDA "Enable CUDA matmul" OFF)

if(USE_CUDA)
  enable_language(CUDA)
  add_library(cuda_kernels STATIC cuda/matmul.cu)
  set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(bench PRIVATE USE_CUDA=1)
  target_link_libraries(bench PRIVATE cuda_kernels)
endif()
