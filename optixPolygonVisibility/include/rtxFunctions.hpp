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

#include "Params.h"
#include "common/common.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <string>
#include <vector>

template <typename IntegerType>
__device__ __host__ IntegerType roundUp(IntegerType x, IntegerType y) {
  return ((x + y - 1) / y) * y;
}

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

/*
 * RTXDataHolder
 *
 * We use this class to pack most of the Optix variables & methods in one place
 * to make it easier to follow the individual steps required to run a specific
 * sample.
 */
struct RTXDataHolder {
  cudaStream_t stream;
  OptixDeviceContext optix_context;
  // Compilation settings for our Module and Pipeline
  OptixModuleCompileOptions moduleCompileOptions = {};
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  // Linking Options
  OptixPipelineLinkOptions pipelineLinkOptions = {};

  // Handle to our traversable (acceleration structure)
  OptixTraversableHandle gas_handle = 0;
  void *d_gas_output_buffer;

  // For simplicity, we work with a single module (ie. single .cu file with
  // optix kernels).
  OptixModule module = 0;
  OptixPipeline pipeline = 0;

  // For simplicity, we just use a minimal number of program groups.
  OptixProgramGroup raygen_group = {};
  OptixProgramGroup miss_group = {};
  OptixProgramGroup hit_group = {};
  OptixShaderBindingTable sbt = {};

  // Log Level 3 gives warnings & errors without too much information
  unsigned int logLevel = 3;

  //==========================================================================
  // Methods
  //==========================================================================
  ~RTXDataHolder();
  void initContext(CUcontext cuCtx = 0);
  void createModule(const std::string ptx_filename);
  void createProgramGroups();
  void linkPipeline();
  void buildSBT();
  void setStream(const cudaStream_t &stream_in);
  void buildAccelerationStructure(std::vector<float3> &vertices,
                                  std::vector<uint3> &triangles);

  OptixAabb read_obj_mesh(const std::string &obj_filename,
                          std::vector<float3> &vertices,
                          std::vector<uint3> &triangles);

  // add ptr to device memory that's going to be freed in the destructor
  void add_allocation(void *ptr) { allocations.push_back(ptr); }

  std::vector<void *> allocations;
};
