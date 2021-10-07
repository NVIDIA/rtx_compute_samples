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

#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "common/common.h"

#include "params.hpp"

template <typename IntegerType>
__device__ __host__ IntegerType roundUp(IntegerType x, IntegerType y) {
  return ((x + y - 1) / y) * y;
}

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct RTXDataHolder {
  Params params;
// set 5 transforms to get same sphere constellation as in spheres5.obj:
  std::vector<Matrix> transforms = {
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, // center
      {1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1}, // 1 1 1
      {1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0}, // 1 0 0
      {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0}, // 0 1 0
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1}  // 0 0 1
  };

  OptixDeviceContext optix_context = nullptr;
  void *d_triangleBuffer;
  OptixTraversableHandle gas_handle;
  OptixTraversableHandle ias_handle;
  CUdeviceptr d_ias_output_buffer = 0;
  void *d_gas_output_buffer;
  OptixModule module = nullptr;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixShaderBindingTable sbt = {};
  OptixProgramGroup raygen_prog_group = nullptr;
  OptixProgramGroup miss_prog_group = nullptr;
  OptixProgramGroup hitgroup_prog_group = nullptr;
  OptixPipeline pipeline = nullptr;
  unsigned int *d_counter;
  int Nrays;
  cudaStream_t stream;

  // Functions to work with Optix
  void initContext();
  void createModule(const std::string ptx_filename);
  void createProgramGroups();
  void linkPipeline();
  void buildSBT();
  void buildIAS();
  OptixAabb buildAccelerationStructure(
     std::vector<std::vector<std::string>> &filesPerBuildInput);
  void setStream(const cudaStream_t &stream_in);
  ~RTXDataHolder();
  void read_obj_mesh(const std::string &obj_filename,
                     std::vector<float3> &vertices,
                     std::vector<uint3> &triangles, OptixAabb &aabb);
};
