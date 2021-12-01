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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "common/common.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

void launch_saxpy_cuda(int N, float a, float *x, float *y);

std::string loadPtx(std::string filename) {
  std::ifstream ptx_in(filename);
  return std::string((std::istreambuf_iterator<char>(ptx_in)),
                     std::istreambuf_iterator<char>());
}

struct SaxpyParameters {
  int N;
  float a;
  float *x;
  float *y;
};

OptixDeviceContext createOptixContext() {
  cudaFree(0); // creates a CUDA context if there isn't already one
  optixInit(); // loads the optix library and populates the function table

  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &optixLogCallback;
  options.logCallbackLevel = 4;

  OptixDeviceContext optix_context = nullptr;
  optixDeviceContextCreate(0, // use current CUDA context
                           &options, &optix_context);

  return optix_context;
}

// load ptx and create module
void loadSaxpyModule(OptixModule &module, OptixDeviceContext optix_context,
                     OptixPipelineCompileOptions &pipeline_compile_options) {
  std::string ptx = loadPtx(BUILD_DIR "/ptx/kernels.ptx");

  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.maxRegisterCount =
      OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

  pipeline_compile_options.usesMotionBlur = false;
  pipeline_compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_compile_options.numPayloadValues = 0;
  pipeline_compile_options.numAttributeValues = 2; // 2 is the minimum
  pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  OPTIX_CHECK(optixModuleCreateFromPTX(optix_context, &module_compile_options,
                                       &pipeline_compile_options, ptx.c_str(),
                                       ptx.size(), nullptr, nullptr, &module));
}

// load ptx and create module
void createSaxpyGroups(OptixProgramGroup *program_groups,
                       OptixDeviceContext optix_context, OptixModule module) {
  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc prog_group_desc[3] = {};

  prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  prog_group_desc[0].raygen.module = module;
  prog_group_desc[0].raygen.entryFunctionName = "__raygen__saxpy";

  // we need to create these but the entryFunctionNames are null
  prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  prog_group_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

  OPTIX_CHECK(optixProgramGroupCreate(optix_context, prog_group_desc, 3,
                                      &program_group_options, nullptr, nullptr,
                                      program_groups));
}

void createSaxpyPipeline(
    OptixPipeline &pipeline, OptixDeviceContext optix_context,
    OptixProgramGroup *program_groups,
    OptixPipelineCompileOptions &pipeline_compile_options) {
  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = 1;
  pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

  OPTIX_CHECK(optixPipelineCreate(optix_context, &pipeline_compile_options,
                                  &pipeline_link_options, program_groups, 3,
                                  nullptr, nullptr, &pipeline));
}

void populateSaxpySBT(OptixShaderBindingTable &sbt,
                      OptixProgramGroup *program_groups) {
  char *device_records;
  CUDA_CHECK(cudaMalloc(&device_records, 3 * OPTIX_SBT_RECORD_HEADER_SIZE));

  char *raygen_record = device_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *miss_record = device_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *hitgroup_record = device_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE;

  char sbt_records[3 * OPTIX_SBT_RECORD_HEADER_SIZE];
  OPTIX_CHECK(optixSbtRecordPackHeader(
      program_groups[0], sbt_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader(
      program_groups[1], sbt_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader(
      program_groups[2], sbt_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE));

  CUDA_CHECK(cudaMemcpy(device_records, sbt_records,
                        3 * OPTIX_SBT_RECORD_HEADER_SIZE,
                        cudaMemcpyHostToDevice));

  sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_record);

  sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record);
  sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  sbt.missRecordCount = 1;

  sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_record);
  sbt.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  sbt.hitgroupRecordCount = 1;
}

int main(int argc, char *argv[]) {
  int N = 1 << 29;
  float a = 2.0f;
  std::vector<float> x(N, 1.0f);
  std::vector<float> y(N, 2.0f);

  // allocate device arrays using regular CUDA
  float *device_x;
  float *device_y;
  CUDA_CHECK(cudaMalloc(&device_x, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc(&device_y, sizeof(float) * N));

  CUDA_CHECK(cudaMemcpy(device_x, x.data(), sizeof(float) * x.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_y, y.data(), sizeof(float) * y.size(),
                        cudaMemcpyHostToDevice));

  // initialize OptiX context
  OptixDeviceContext optix_context = createOptixContext();

  // load ptx and create module
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixModule module = nullptr;
  loadSaxpyModule(module, optix_context, pipeline_compile_options);

  // creat program groups
  OptixProgramGroup program_groups[3] = {};
  createSaxpyGroups(program_groups, optix_context, module);

  // assemble program groups into pipeline
  OptixPipeline pipeline = nullptr;
  createSaxpyPipeline(pipeline, optix_context, program_groups,
                      pipeline_compile_options);

  // setup shader binding table
  OptixShaderBindingTable sbt = {};
  populateSaxpySBT(sbt, program_groups);

  // populate and move parameters to device
  SaxpyParameters params{N, a, device_x, device_y};
  SaxpyParameters *device_params;
  CUDA_CHECK(cudaMalloc(&device_params, sizeof(SaxpyParameters)));
  CUDA_CHECK(cudaMemcpy(device_params, &params, sizeof(SaxpyParameters),
                        cudaMemcpyHostToDevice));

  // regular cuda kernel saxpy
  launch_saxpy_cuda(N, 2.0f, device_x, device_y);

  // restore y
  CUDA_CHECK(cudaMemcpy(device_y, y.data(), sizeof(float) * y.size(),
                        cudaMemcpyHostToDevice));

  // optix launch saxpy
  OPTIX_CHECK(optixLaunch(pipeline, 0,
                          reinterpret_cast<CUdeviceptr>(device_params),
                          sizeof(SaxpyParameters), &sbt, N, 1, 1));

  // copy back data
  CUDA_CHECK(cudaMemcpy(y.data(), device_y, sizeof(float) * y.size(),
                        cudaMemcpyDeviceToHost));

  // check for differences
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = std::max(maxError, std::abs(y[i] - 4.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // clean up
  OPTIX_CHECK(optixPipelineDestroy(pipeline));
  for (int i = 0; i < 3; ++i) {
    OPTIX_CHECK(optixProgramGroupDestroy(program_groups[i]));
  }
  OPTIX_CHECK(optixModuleDestroy(module));
  OPTIX_CHECK(optixDeviceContextDestroy(optix_context));

  CUDA_CHECK(cudaFree(device_params));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
  CUDA_CHECK(cudaFree(device_y));
  CUDA_CHECK(cudaFree(device_x));

  return 0;
}
