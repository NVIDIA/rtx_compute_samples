cmake_minimum_required(VERSION 3.5)

project(optix_meshrefine LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

cuda_compile_ptx(
  cuda_ptx_files
  camera.cu
  OPTIONS "-I${OPTIX_HOME}/include"
  )

# rename ptx file
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/camera.ptx COMMAND ${CMAKE_COMMAND} -E rename ${cuda_ptx_files} ${PROJECT_BINARY_DIR}/camera.ptx DEPENDS ${cuda_ptx_files})


add_custom_target(meshrefine_ptx ALL
  DEPENDS ${PROJECT_BINARY_DIR}/camera.ptx camera.cu
  SOURCES camera.cu
  )

add_executable(meshrefine
  optixMeshRefine.cpp
  )

target_link_libraries(meshrefine
  "${OPTIX_HOME}/lib64/liboptix.so"
  ${CUDA_LIBRARIES}
  )

target_include_directories(meshrefine
  PRIVATE
  "${OPTIX_HOME}/include"
  ${CUDA_INCLUDE_DIRS}
  )

# copy sphere obj
add_custom_command(TARGET meshrefine POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/sphere.obj ${PROJECT_BINARY_DIR}/sphere.obj)
