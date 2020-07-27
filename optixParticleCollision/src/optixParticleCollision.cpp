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
#include "utils.h"

void print_usage(std::string const &cmd) {
  std::cout << "Usage: \n"
               "\t"
            << cmd << " [particle_count] [output.csv]\n\n";
}

// Dumps particle data into a csv file for simple visualization
void write_csv(std::string const &path, float3 *particle_positions,
               size_t num_particles) {
  std::cout << "Writing output to '" + path + "'\n";
  std::ofstream file(path);
  if (!file)
    throw std::runtime_error("Could not open file for writing: '" + path + "'");

  // Write out all particles
  for (size_t i = 0; i < num_particles; ++i) {
    file << std::setprecision(16) << particle_positions[i].x << '\t'
         << particle_positions[i].y << '\t' << particle_positions[i].z << '\n';
  }
}

int main(int argc, char *argv[]) try {
  // Check program arguments
  if ((argc > 1 && std::string(argv[1]) == "-h") ||
      (argc > 1 && std::string(argv[1]) == "--help") || (argc > 3)) {
    print_usage(argv[0]);
    return 1;
  }

  std::string obj_file = OBJ_DIR "cow.obj";
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
  std::vector<float3> vertices;
  std::vector<uint3> triangles;
  OptixAabb aabb_box =
      rtx_dataholder->buildAccelerationStructure(obj_file, vertices, triangles);

  //==========================================================================
  // Prepare Launch
  //==========================================================================
  // Get the number of particles from the argument
  size_t const num_particles =
      (argc > 1) ? std::max(1, std::atoi(argv[1])) : 1000000;
  std::cout << std::setw(32) << std::left
            << "Number of Particles: " << std::setw(12) << std::right
            << num_particles << "\n";

  // Initialize Particle's Position & Velocity
  initialize_particles(num_particles, aabb_box,
                       &(g_pipeline_params.particle_positions),
                       &(g_pipeline_params.particle_velocities));

  // Copy Pipeline Parameters to Device Memory
  PipelineParams *d_pipeline_params = {};
  CUDA_CHECK(cudaMalloc((void **)&d_pipeline_params, sizeof(PipelineParams)));
  CUDA_CHECK(cudaMemcpy(d_pipeline_params, &g_pipeline_params,
                        sizeof(PipelineParams), cudaMemcpyHostToDevice));

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
        reinterpret_cast<CUdeviceptr>(d_pipeline_params),
        sizeof(PipelineParams), // Pass our Pipeline Parameters so we can access
                                // them on the Device
        &sbt,                   // Pass the SBT
        num_particles, 1,
        1 // x, y, z launch dimensions. For our problem, 1D is just fine
        ));
    // We synchronize here for the timer
    cudaStreamSynchronize(stream);
    timer.stop();

    // Print the Stats
    std::cout << std::setw(32) << std::left << "Launch took: " << std::setw(12)
              << std::right << timer.get_elapsed_s() << "s\n";
    std::cout << std::setw(32) << std::left
              << "Updates & rays per second: " << std::setw(12) << std::right
              << num_particles / timer.get_elapsed_s() << "\n\n";
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  // For dumping a csv file with the final particle positions
  if (argc > 2) {
    std::string csv_path{argv[2]};

    // Copy particle positions from device back to host
    std::vector<float3> final_particle_positions;
    final_particle_positions.resize(num_particles);
    CUDA_CHECK(cudaMemcpy(
        final_particle_positions.data(), g_pipeline_params.particle_positions,
        sizeof(float3) * num_particles, cudaMemcpyDeviceToHost));
    write_csv(csv_path, final_particle_positions.data(), num_particles);
  }

  cudaFree(d_pipeline_params);
  cudaFree(g_pipeline_params.particle_positions);
  cudaFree(g_pipeline_params.particle_velocities);

  return 0;
} catch (std::exception const &e) {
  std::cerr << "Caught Exception: '" << e.what() << "'\n";
}
