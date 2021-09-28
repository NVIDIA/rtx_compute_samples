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

#include "maxBounce.h"
#include "params.hpp"
#include "rtxFunctions.hpp"


RTXDataHolder *rtx_dataholder;

// small problem for debugging
uint32_t width = 8u;  // buffer size x
uint32_t height = 8u; // buffer size y
uint32_t depth = 1u;

// fits on a single screen
// uint32_t height = 32u; // buffer size y
// uint32_t width = 32u;  // buffer size x

// 100M rays
// uint32_t  height = 10000u;   // buffer size y
// uint32_t  width  = 10000u;   // buffer size x

int main(int argc, char **argv) {

  // std::vector<std::vector<std::string>> buildInputsFileName = { {"planes0", "planes1"},  {"sphere"} };
 std::vector<std::vector<std::string>> buildInputsFileName = { {"planes0", "planes1" ,  "sphere" }};
  for(int i= 0; i< buildInputsFileName.size() ; ++i ){
    for( int j = 0 ; j< buildInputsFileName[i].size(); ++j) {
      std::string obj_file = OBJ_DIR "" + buildInputsFileName[i][j] + ".obj";
      buildInputsFileName[i][j] = obj_file;
      std::cout << "Adding mesh file = " << obj_file << " to buildInput "<< i << std::endl;
      }
  }
 
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

 
  std::cout << "Building Acceleration Structure \n";
  OptixAabb aabb_box = rtx_dataholder->buildAccelerationStructure(buildInputsFileName);

  // calculate delta
  float3 delta = make_float3((aabb_box.maxX - aabb_box.minX) / width,
                             (aabb_box.maxY - aabb_box.minY) / height,
                             (aabb_box.maxZ - aabb_box.minZ) / depth);
  float3 min_corner = make_float3(aabb_box.minX, aabb_box.minY, aabb_box.minZ);

  float *d_tpath;
  CUDA_CHECK(cudaMalloc((void **)&d_tpath, width * height * sizeof(float)));

    uint32_t *d_planeHits;
  CUDA_CHECK(cudaMalloc((void **)&d_planeHits, 1* sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_planeHits, 0,1*sizeof(uint32_t)));

  uint32_t *d_sphereHits;
  CUDA_CHECK(cudaMalloc((void **)&d_sphereHits, 1* sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_sphereHits, 0,1*sizeof(uint32_t)));



  // Algorithmic parameters and data pointers used in GPU program
  Params params;
  params.min_corner = min_corner;
  params.delta = delta;
  params.handle = rtx_dataholder->gas_handle;
  params.width = width;
  params.height = height;
  params.depth = depth;
  params.tpath = d_tpath; 
  params.sphereHitCounter =  d_sphereHits;
  params.planeHitCounter =  d_planeHits;


  Params *d_param;
  CUDA_CHECK(cudaMalloc((void **)&d_param, sizeof(Params)));
  CUDA_CHECK(
      cudaMemcpy(d_param, &params, sizeof(params), cudaMemcpyHostToDevice));

  const OptixShaderBindingTable &sbt = rtx_dataholder->sbt;

  std::cout << "Launching Ray Tracer to scatter rays \n";
  OPTIX_CHECK(optixLaunch(rtx_dataholder->pipeline, stream,
                          reinterpret_cast<CUdeviceptr>(d_param),
                          sizeof(Params), &sbt, width, height, depth));

  CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t sphereHits = 0;
  CUDA_CHECK(cudaMemcpy(&sphereHits, d_sphereHits , 1*sizeof(uint32_t), cudaMemcpyDeviceToHost));


  uint32_t planeHits = 0;
  CUDA_CHECK(cudaMemcpy(&planeHits, d_planeHits , 1*sizeof(uint32_t), cudaMemcpyDeviceToHost));
 
  std::cout<< " launched  "<< width*height << " rays from corner ("<< min_corner.x << "," << min_corner.y<<"," << min_corner.z<<   ") of which " << sphereHits << " ray hit the sphere and "<< planeHits << " ray hit the planes  \n";

  std::cout << "Cleaning up ... \n";
  CUDA_CHECK(cudaFree(d_tpath));
  CUDA_CHECK(cudaFree(d_param));
  CUDA_CHECK(cudaFree(d_planeHits));
  CUDA_CHECK(cudaFree(d_sphereHits));
  delete rtx_dataholder;

  return 0;
}
