cmake_minimum_required(VERSION 3.5)

project(optix_uniqpaths LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

cuda_compile_ptx(
  cuda_ptx_files
  optixUniqPathsPrograms.cu
  OPTIONS "-I${OPTIX_HOME}/include"
  )

# rename ptx file
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/optixUniqPathsPrograms.ptx COMMAND ${CMAKE_COMMAND} -E rename ${cuda_ptx_files} ${PROJECT_BINARY_DIR}/optixUniqPathsPrograms.ptx DEPENDS ${cuda_ptx_files})


add_custom_target(uniqpaths_ptx ALL
  DEPENDS ${PROJECT_BINARY_DIR}/optixUniqPathsPrograms.ptx optixUniqPathsPrograms.cu
  SOURCES optixUniqPathsPrograms.cu
  )

add_executable(uniqpaths
  optixUniqPaths.cpp
  thrustSortPaths
  )

target_link_libraries(uniqpaths
  "${OPTIX_HOME}/lib64/liboptix.so"
  ${CUDA_LIBRARIES}
  )

target_include_directories(uniqpaths
  PRIVATE
  "${OPTIX_HOME}/include"
  ${CUDA_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  )

# copy sphere obj
add_custom_command(TARGET uniqpaths POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/sphereAndPlane.obj ${PROJECT_BINARY_DIR}/sphereAndPlane.obj)
