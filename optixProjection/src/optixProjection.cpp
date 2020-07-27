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

#include <iostream>

#include "common/common.h"

#include "params.hpp"
#include "rtxFunctions.hpp"

RTXDataHolder *rtx_dataholder;
uint32_t width = 8u;
uint32_t height = 8u;
uint32_t depth = 8u;

int main(int argc, char **argv) {
  std::string obj_file = OBJ_DIR "cow.obj";
  std::string ptx_filename = BUILD_DIR "/ptx/optixPrograms.ptx";
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  rtx_dataholder = new RTXDataHolder();
  std::cout << "Initializing Context \n";
  rtx_dataholder->initContext();
  std::cout << "Reading PTX file and creating modules \n";
  rtx_dataholder->createModule(ptx_filename);
  std::cout << "Creating Optix Program Groups \n";
  rtx_dataholder->createProgramGroups();
  std::cout << "Linking Pipeline \n";
  rtx_dataholder->linkPipeline();
  std::cout << "Building Shader Binding Table (SBT) \n";
  rtx_dataholder->buildSBT();

  std::vector<float3> vertices;
  std::vector<uint3> triangles;
  std::cout << "Building Acceleration Structure \n";
  OptixAabb aabb_box =
      rtx_dataholder->buildAccelerationStructure(obj_file, vertices, triangles);

  // calculate delta
  float3 delta = make_float3((aabb_box.maxX - aabb_box.minX) / width,
                             (aabb_box.maxY - aabb_box.minY) / height,
                             (aabb_box.maxZ - aabb_box.minZ) / depth);
  float3 min_point = make_float3(aabb_box.minX, aabb_box.minY, aabb_box.minZ);
  float3 far_away_point =
      make_float3(min_point.x - 1, min_point.y - 1, min_point.z - 1);

  float *d_output;
  CUDA_CHECK(cudaMalloc((void **)&d_output, width * height * sizeof(float)));

  // Algorithmic parameters and data pointers used in GPU program
  Params params;
  params.min_point = min_point;
  params.far_away_point = far_away_point;
  params.delta = delta;
  params.handle = rtx_dataholder->gas_handle;
  params.width = width;
  params.height = height;
  params.depth = depth;
  params.output = d_output;

  Params *d_param;
  CUDA_CHECK(cudaMalloc((void **)&d_param, sizeof(Params)));
  CUDA_CHECK(
      cudaMemcpy(d_param, &params, sizeof(params), cudaMemcpyHostToDevice));

  const OptixShaderBindingTable &sbt = rtx_dataholder->sbt;

  std::cout << "Launching Ray Tracer to compute Projections \n";
  OPTIX_CHECK(optixLaunch(rtx_dataholder->pipeline, stream,
                          reinterpret_cast<CUdeviceptr>(d_param),
                          sizeof(Params), &sbt, width, height, 1));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::cout << "Cleaning up ... \n";
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_param));
  delete rtx_dataholder;

  return 0;
}
