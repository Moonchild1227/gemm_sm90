cmake_minimum_required(VERSION 3.24)
project(multi_stage_gemm LANGUAGES CXX)

option(ENABLE_CUDA "Build CUDA targets" ON)
option(ENABLE_PYBIND "Build pybind11 extension" ON)

set(CUTLASS_ROOT "${CMAKE_SOURCE_DIR}/thirdpart/cutlass" CACHE PATH "Path to CUTLASS root")

if(ENABLE_CUDA)
  add_subdirectory(csrc)
else()
  message(STATUS "ENABLE_CUDA=OFF: skipping CUDA targets.")
endif()

