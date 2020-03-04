cmake_minimum_required(VERSION 3.5)

project(optix_saxpy LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

cuda_compile_ptx(
  cuda_ptx_files
  kernels.cu
  OPTIONS "-I${OPTIX_HOME}/include"
  )

add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/kernels.ptx COMMAND ${CMAKE_COMMAND} -E rename ${cuda_ptx_files} ${PROJECT_BINARY_DIR}/kernels.ptx DEPENDS ${cuda_ptx_files}) 
add_custom_target(ptx ALL
  DEPENDS ${PROJECT_BINARY_DIR}/kernels.ptx kernels.cu
  SOURCES kernels.cu
  )

add_executable(saxpy
  saxpy_optix.cpp
  cuda.cu
  )

target_link_libraries(saxpy
  "${OPTIX_HOME}/lib64/liboptix.so"
  )

target_include_directories(saxpy
  PRIVATE
  "${OPTIX_HOME}/include"
  ${CUDA_INCLUDE_DIRS}
  )
