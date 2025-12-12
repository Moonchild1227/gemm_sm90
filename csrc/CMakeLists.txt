cmake_minimum_required(VERSION 3.24)

if(NOT ENABLE_CUDA)
  message(STATUS "CUDA disabled; skipping csrc targets.")
  return()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING "CUDA compiler not found. Configure on a CUDA host or set CUDACXX.")
  return()
endif()
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)

find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
  message(WARNING "CUDAToolkit not found; CUDA targets will be skipped.")
  return()
endif()

# CUTLASS/CUTE headers
if(NOT EXISTS "${CUTLASS_ROOT}/include")
  message(FATAL_ERROR "CUTLASS_ROOT (${CUTLASS_ROOT}) does not contain include/.")
endif()

# pybind11 optional: if not found and no include hint, skip building the module
set(PYBIND_OK OFF)
if(ENABLE_PYBIND)
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    set(PYBIND_OK ON)
  else()
    find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h)
    if(PYBIND11_INCLUDE_DIR)
      set(PYBIND_OK ON)
    else()
      message(WARNING "pybind11 not found; disable pybind target. Install pybind11 or set PYBIND11_INCLUDE_DIR.")
    endif()
  endif()
endif()

# Object library for the kernel to allow reuse.
add_library(gemm_kernel_obj OBJECT kernel.cu)
target_include_directories(gemm_kernel_obj
  PUBLIC
    ${CUTLASS_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(gemm_kernel_obj PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
set_target_properties(gemm_kernel_obj PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

if(PYBIND_OK)
  add_library(multi_stage_gemm MODULE pybind_wrapper.cpp $<TARGET_OBJECTS:gemm_kernel_obj>)
  target_include_directories(multi_stage_gemm
    PUBLIC
      ${CUTLASS_ROOT}/include
      ${CUDAToolkit_INCLUDE_DIRS}
  )
  if(pybind11_FOUND)
    target_link_libraries(multi_stage_gemm PRIVATE pybind11::module)
  else()
    target_include_directories(multi_stage_gemm PRIVATE ${PYBIND11_INCLUDE_DIR})
  endif()
  target_link_libraries(multi_stage_gemm PRIVATE CUDA::cudart)
  set_target_properties(multi_stage_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""  # for Python extension naming
  )
else()
  message(STATUS "pybind target skipped (pybind11 not found).")
endif()

