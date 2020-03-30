cmake_minimum_required(VERSION 3.5)

project(optix_compute_sample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

# ensure OPTIX
if(NOT OPTIX_HOME)
  if(NOT DEFINED ENV{OPTIX_HOME})
    message(FATAL_ERROR "OPTIX_HOME not defined. Please set OPTIX_HOME environment variable or -DOPTIX_HOME=DIR via CMake")
  else()
    set(OPTIX_HOME "$ENV{OPTIX_HOME}" CACHE PATH "Location of OPTIX SDK")
  endif()
endif()

add_subdirectory(optixMeshRefine)
add_subdirectory(optixProjection)
add_subdirectory(optixRayScattering)
add_subdirectory(optixSaxpy)
add_subdirectory(optixUniqPaths)
add_subdirectory(optixVolumeSampling)
add_subdirectory(optixParticleCollision)
