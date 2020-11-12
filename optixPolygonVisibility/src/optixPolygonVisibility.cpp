/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Standard Headers
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

// CUDA Headers
#include <cuda_runtime.h>

// Optix Headers
#include <optix.h>
#include <optix_stubs.h>
// Use this in your main compilation unit
#include <optix_function_table_definition.h>

// Local Headers
#include "common/Timer.h"
#include "common/common.h"

#include "Params.h"
#include "rtxFunctions.hpp"
#include "utils.cpp"

int main(int argc, char *argv[]) try {
  // Check program arguments
  if ((argc > 1 && std::string(argv[1]) == "-h") ||
      (argc > 1 && std::string(argv[1]) == "--help") || (argc > 4)) {
    print_usage(argv[0]);
    return 1;
  }

  // Get the number of polygons and points from the argument
  size_t const num_polygons =
      (argc > 1) ? std::max(1, std::atoi(argv[1])) : 1e4;
  size_t const num_centers = (argc > 1) ? std::max(1, std::atoi(argv[2])) : 1e6;
  const int N_rays =
      360 * num_centers; // Each point does a 360 launch. 1 degree sampling

  std::cout << std::setw(32) << std::left
            << "Number of Polygons: " << std::setw(12) << std::right
            << num_polygons << "\n";
  std::cout << std::setw(32) << std::left
            << "Number of Ray Centers: " << std::setw(12) << std::right
            << num_centers << "\n";
  std::cout << std::setw(32) << std::left << "Number of Rays: " << std::setw(12)
            << std::right << N_rays << "\n";

  // std::string obj_file = OBJ_DIR "sq3d.obj";
  std::string ptx_filename = BUILD_DIR "/ptx/optixPrograms.ptx";

  RTXDataHolder *rtx_dataholder = new RTXDataHolder();
  // Initialize CUDA & Optix Context
  rtx_dataholder->initContext();
  // Compile an Optix Module from PTX code
  rtx_dataholder->createModule(ptx_filename);
  // Create Program Groups (RayHit, Miss, RayGen) from the Module
  rtx_dataholder->createProgramGroups();
  // Link Program Groups into an Optix Pipeline
  rtx_dataholder->linkPipeline();
  // Build the SBT
  rtx_dataholder->buildSBT();

  // Build Acceleration structure

  std::vector<float2> vertices;
  std::vector<uint3> triangles;
  float3 min = make_float3(-100.0f, -100.0f, 0.0f);
  float3 max = make_float3(100.0f, 100.0f, 0.0f);
  build_2d_geometry(min, max, num_polygons, vertices, triangles);

  // this function raises each triangle in 2d by delta in z direction. Hence for
  // every edge in 2D, two triangles are created in 3D
  std::vector<float3> vertices3d;
  std::vector<uint3> triangles3d;
  float delta = 0.1;
  OptixAabb aabb_box = elevate_2d_to_3d(num_polygons, vertices, triangles,
                                        vertices3d, triangles3d, delta);

  rtx_dataholder->buildAccelerationStructure(vertices3d, triangles3d);

  std::cout << "num of triangles --> " << triangles.size() << std::endl;
  std::cout << "num of vertices --> " << vertices.size() << std::endl;

  //==========================================================================
  // Prepare Launch
  //==========================================================================

  // Initialize Particle's Position & Velocity
  float3 *ray_centers;
  initialize_raycenters(num_centers, aabb_box, &ray_centers);

  // Allocate data for storing hit points.
  float *out_tmax;
  CUDA_CHECK(cudaMalloc((void **)&out_tmax, N_rays * sizeof(float)));
  CUDA_CHECK(cudaMemset((void *)out_tmax, 0, N_rays * sizeof(float)));

  // Algorithmic parameters and data pointers used in GPU program
  Params params;
  params.handle = rtx_dataholder->gas_handle;
  params.ray_centers = ray_centers;
  params.hit_tmax = out_tmax;

  // Copy Pipeline Parameters to Device Memory
  Params *d_params = {};
  CUDA_CHECK(cudaMalloc((void **)&d_params, sizeof(Params)));
  CUDA_CHECK(
      cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));

  // Create a CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  {
    // We synchronize here for the timer
    cudaStreamSynchronize(stream);
    Timer timer;

    const OptixShaderBindingTable &sbt = rtx_dataholder->sbt;

    // Launch Optix! Yeeeey!
    OPTIX_CHECK(optixLaunch(
        rtx_dataholder
            ->pipeline, // This is our pipeline with all the program groups
        stream,         // Can define a CUDA stream to run in
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(Params), // Pass our Pipeline Parameters so we can access
                        // them on the Device
        &sbt,           // Pass the SBT
        N_rays, 1,
        1 // x, y, z launch dimensions. For our problem, 1D is just fine
        ));
    // We synchronize here for the timer
    cudaStreamSynchronize(stream);
    timer.stop();

    // Print the Stats
    std::cout << std::setw(32) << std::left << "Launch took: " << std::setw(12)
              << std::right << timer.get_elapsed_s() << "s\n";
    std::cout << std::setw(32) << std::left
              << "rays per second: " << std::setw(12) << std::right
              << N_rays / timer.get_elapsed_s() << "\n\n";
  }

  // For dumping a csv file with the final particle positions
  if (argc > 3) {
    std::string csv_path{argv[3]};

    // Copy particle positions from device back to host
    std::vector<float> hit_tmax;
    hit_tmax.resize(N_rays);
    CUDA_CHECK(cudaMemcpy(hit_tmax.data(), out_tmax, sizeof(float) * N_rays,
                          cudaMemcpyDeviceToHost));
    write_csv(csv_path, hit_tmax.data(), N_rays);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  cudaFree(d_params);
  cudaFree(ray_centers);
  cudaFree(out_tmax);

  return 0;
} catch (std::exception const &e) {
  std::cerr << "Caught Exception: '" << e.what() << "'\n";
}
