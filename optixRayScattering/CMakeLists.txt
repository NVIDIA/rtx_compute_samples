cmake_minimum_required(VERSION 3.5)

project(optix_rayscattering LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

cuda_compile_ptx(
  cuda_ptx_files
  optixRayScatteringPrograms.cu
  OPTIONS "-I${OPTIX_HOME}/include"
  )

# rename ptx file
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/optixRayScatteringPrograms.ptx COMMAND ${CMAKE_COMMAND} -E rename ${cuda_ptx_files} ${PROJECT_BINARY_DIR}/optixRayScatteringPrograms.ptx DEPENDS ${cuda_ptx_files})


add_custom_target(rayscattering_ptx ALL
  DEPENDS ${PROJECT_BINARY_DIR}/optixRayScatteringPrograms.ptx optixRayScatteringPrograms.cu
  SOURCES optixRayScatteringPrograms.cu
  )

add_executable(rayscattering
  optixRayScattering.cpp
  )

target_link_libraries(rayscattering
  "${OPTIX_HOME}/lib64/liboptix.so"
  ${CUDA_LIBRARIES}
  )

target_include_directories(rayscattering
  PRIVATE
  "${OPTIX_HOME}/include"
  ${CUDA_INCLUDE_DIRS}
  )

# copy sphere obj
add_custom_command(TARGET rayscattering POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/sphereAndPlane.obj ${PROJECT_BINARY_DIR}/sphereAndPlane.obj)
