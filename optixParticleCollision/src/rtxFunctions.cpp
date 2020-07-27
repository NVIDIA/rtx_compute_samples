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

#include "rtxFunctions.hpp"
#include "common/common.h"
#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "common/common.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"

RTXDataHolder::~RTXDataHolder() {
  // Cleanup Procedure in reverse order
  optixPipelineDestroy(pipeline);
  optixProgramGroupDestroy(raygen_group);
  optixProgramGroupDestroy(miss_group);
  optixProgramGroupDestroy(hit_group);
  optixModuleDestroy(module);

  for (auto ptr : allocations)
    CUDA_CHECK(cudaFree(ptr));
}

void RTXDataHolder::initContext(CUcontext cuCtx) {
  // Make sure everyone knows when they're running a Debug build
#ifndef NDEBUG
  std::cout << "This is a Debug Build!\n";
#endif
  // Print active CUDA devices to make it clear what devices the sample is
  // running on. Use the CUDA_VISIBLE_DEVICES environment variable to select
  // which ones to run on.
  printActiveCudaDevices();

  // Make sure CUDA is initialized by making a 'dummy' call
  CUDA_CHECK(cudaFree(0));

  // Initialize Optix
  OPTIX_CHECK(optixInit());

  // Create Optix Context
  OPTIX_CHECK(
      optixDeviceContextCreate(cuCtx,   // 0 means default CUDA context
                               nullptr, // not providing any context options
                               &optix_context));

  // Register Logging Callback
  optixDeviceContextSetLogCallback(optix_context, optixLogCallback, nullptr,
                                   logLevel);
}

void RTXDataHolder::createModule(const std::string ptx_filename) {
  // Set up module compilation options
  // This is stored in the RTXDataHolder to ensure consistent settings across
  // potentially multiple module compilations.
  {
    moduleCompileOptions.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    // We use different settings for Release & Debug builds
#ifndef NDEBUG
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel =
        OPTIX_COMPILE_DEBUG_LEVEL_NONE; // TODO: or is LINEINFO ok?
#endif
  }

  // Set up pipeline compilation options
  // This is stored in the RTXDataHolder to ensure consistent settings across
  // potentially multiple module compilations.
  {
    pipelineCompileOptions.usesMotionBlur = 0;
    // Highlight that we're only tracing against a single GAS as this might
    // enable additional optimizations
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    // Number of Payloads used in optixTrace on the Device
    pipelineCompileOptions.numPayloadValues = 1;
    // 2 Attributes are the minimum
    pipelineCompileOptions.numAttributeValues = 2;
    // Name of the global variable in constant memory which receives the
    // pipeline parameters from the optixLaunch call
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
#ifndef NDEBUG
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
#else
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  }

  //==========================================================================
  // Create Module from PTX
  //==========================================================================
  std::string const ptxContent = getFileContent(ptx_filename);
  OPTIX_CHECK(optixModuleCreateFromPTX(
      optix_context, &moduleCompileOptions, &pipelineCompileOptions,
      ptxContent.c_str(), ptxContent.size(), nullptr, 0, &module));
}

void RTXDataHolder::createProgramGroups() {
  // A program group ties a certain function (or functions for hit groups) in
  // our Optix module to a specific ray event (or callables).
  // We need the program groups to populate the shader binding table.

  //==========================================================================
  // Build Program Groups
  //==========================================================================
  std::vector<OptixProgramGroupDesc> programGroupDescs;
  OptixProgramGroupDesc cur_desc;

  // Ray Generation Program Group
  cur_desc = {};
  cur_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  cur_desc.raygen.module = module;
  cur_desc.raygen.entryFunctionName = "__raygen__camera";
  programGroupDescs.push_back(cur_desc);

  // Miss Program Group
  // We don't need a miss program but we still need a miss program group for
  // our SBT. So we create a program group for the miss kind but leave the
  // module and entryFunctionName empty.
  // This is more efficient than writing an empty miss function!
  cur_desc = {};
  cur_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  programGroupDescs.push_back(cur_desc);

  // Ray Hit Program Group
  cur_desc = {};
  cur_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  cur_desc.hitgroup.moduleCH = module;
  cur_desc.hitgroup.entryFunctionNameCH = "__closesthit__";
  programGroupDescs.push_back(cur_desc);

  // We create all three program groups at once
  OptixProgramGroupOptions programGroupOptions = {};
  std::vector<OptixProgramGroup> program_groups(programGroupDescs.size());
  OPTIX_CHECK(optixProgramGroupCreate(
      optix_context, programGroupDescs.data(), programGroupDescs.size(),
      &programGroupOptions, nullptr, 0, program_groups.data()));

  raygen_group = program_groups[0];
  miss_group = program_groups[1];
  hit_group = program_groups[2];
}

void RTXDataHolder::linkPipeline() {
  // Gather all the program groups we need in our pipeline
  std::vector<OptixProgramGroup> pipeline_program_groups;
  pipeline_program_groups.push_back(raygen_group);
  pipeline_program_groups.push_back(miss_group);
  pipeline_program_groups.push_back(hit_group);

  OptixPipelineLinkOptions pipelineLinkOptions = {};

  // We don't need recursive tracing in this example. If you want to simulate
  // scattering events, You'd need to increase this to the maximum  number of
  // recursive steps.
  pipelineLinkOptions.maxTraceDepth = 1;
#ifndef NDEBUG
  pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
  pipelineLinkOptions.debugLevel =
      OPTIX_COMPILE_DEBUG_LEVEL_NONE; // TODO: or is LINEINFO ok?
#endif
  // Create our pipeline
  OPTIX_CHECK(optixPipelineCreate(
      optix_context, &pipelineCompileOptions, &pipelineLinkOptions,
      pipeline_program_groups.data(), pipeline_program_groups.size(), nullptr,
      0, &pipeline));

  // We set the stack size to be minimal. You might want to increase this once
  // your build on top of this sample.
  // OPTIX_CHECK( optixPipelineSetStackSize(pipeline, 0, 0, 0, 1));
}

void RTXDataHolder::buildSBT() {
  // We keep our SBT as minimal as possible. This means that we're always using
  // an SBT stride and offset of 0 in our trace calls, we do not use instances
  // with SBT offsets, etc.
  //
  // With this setup, we only need to provide a ray generation program, a miss
  // program (can be an empty program group) and a ray hit program.
  //
  // The SBT is populated not with programs but with SBT records which consist
  // of a Optix-managed header and a (optional) user-provided data section.
  //
  // The header contains information about what program group to call and is
  // filled through optixSbtRecordPackHeader

  using EmptyRecord = SbtRecord<void>;
  EmptyRecord emptyRecord;

  // Raygen
  {
    // Fill Header of Shader Binding Table Record
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_group, &emptyRecord));
    void *d_sbt;
    CUDA_CHECK(cudaMalloc(&d_sbt, sizeof(emptyRecord)));
    CUDA_CHECK(cudaMemcpy(d_sbt, &emptyRecord, sizeof(emptyRecord),
                          cudaMemcpyHostToDevice));
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(d_sbt);
    allocations.push_back(d_sbt);
  }

  // Miss
  {
    // Fill Header of Shader Binding Table Record
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_group, &emptyRecord));
    void *d_sbt;
    CUDA_CHECK(cudaMalloc(&d_sbt, sizeof(emptyRecord)));
    CUDA_CHECK(cudaMemcpy(d_sbt, &emptyRecord, sizeof(emptyRecord),
                          cudaMemcpyHostToDevice));
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(d_sbt);
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    allocations.push_back(d_sbt);
  }

  // Hit Group
  {
    // Fill Header of Shader Binding Table Record
    OPTIX_CHECK(optixSbtRecordPackHeader(hit_group, &emptyRecord));
    void *d_sbt;
    CUDA_CHECK(cudaMalloc(&d_sbt, sizeof(emptyRecord)));
    CUDA_CHECK(cudaMemcpy(d_sbt, &emptyRecord, sizeof(emptyRecord),
                          cudaMemcpyHostToDevice));
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_sbt);
    sbt.hitgroupRecordCount = 1;
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    allocations.push_back(d_sbt);
  }
}

/**
 * Builds an accleration structure from an OBJ object. Computes its bounding box
 * and stores it in \p bbox. A simple geometric acceleration structure (GAS) is
 * constructed and \p handle is set to its handle.
 */
OptixAabb
RTXDataHolder::buildAccelerationStructure(const std::string obj_filename,
                                          std::vector<float3> &vertices,
                                          std::vector<uint3> &triangles) {
  OptixAabb aabb;

  aabb = read_obj_mesh(obj_filename, vertices, triangles);
  float3 *d_vertices;
  const size_t vertices_size = sizeof(float3) * vertices.size();
  CUDA_CHECK(cudaMalloc(&d_vertices, vertices_size));
  CUDA_CHECK(cudaMemcpy(d_vertices, vertices.data(), vertices_size,
                        cudaMemcpyHostToDevice));

  const size_t tri_size = sizeof(uint3) * triangles.size();
  void *d_triangles;
  CUDA_CHECK(cudaMalloc(&d_triangles, tri_size));
  CUDA_CHECK(cudaMemcpy(d_triangles, triangles.data(), tri_size,
                        cudaMemcpyHostToDevice));

  const unsigned int triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
  OptixBuildInput triangle_input = {};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.numVertices =
      static_cast<unsigned int>(vertices.size());
  triangle_input.triangleArray.vertexBuffers =
      reinterpret_cast<CUdeviceptr *>(&d_vertices);
  triangle_input.triangleArray.flags = triangle_input_flags;
  triangle_input.triangleArray.numSbtRecords = 1;
  triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangle_input.triangleArray.numIndexTriplets =
      static_cast<unsigned int>(triangles.size());
  triangle_input.triangleArray.indexBuffer =
      reinterpret_cast<CUdeviceptr>(d_triangles);

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context, &accel_options,
                                           &triangle_input,
                                           1, // Number of build input
                                           &gas_buffer_sizes));
  void *d_temp_buffer_gas;
  CUDA_CHECK(cudaMalloc(&d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes));

  // non-compacted output
  void *d_buffer_temp_output_gas_and_compacted_size;
  size_t compactedSizeOffset =
      roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CUDA_CHECK(cudaMalloc(&d_buffer_temp_output_gas_and_compacted_size,
                        compactedSizeOffset + 8));

  OptixAccelEmitDesc emitProperty = {};
  emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitProperty.result =
      (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size +
                    compactedSizeOffset);

  OPTIX_CHECK(optixAccelBuild(optix_context,
                              0, // CUDA stream
                              &accel_options, &triangle_input,
                              1, // num build inputs
                              reinterpret_cast<CUdeviceptr>(d_temp_buffer_gas),
                              gas_buffer_sizes.tempSizeInBytes,
                              reinterpret_cast<CUdeviceptr>(
                                  d_buffer_temp_output_gas_and_compacted_size),
                              gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                              &emitProperty, // emitted property list
                              1              // num emitted properties
                              ));

  CUDA_CHECK(cudaFree(d_temp_buffer_gas));

  size_t compacted_gas_size;
  CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result,
                        sizeof(size_t), cudaMemcpyDeviceToHost));

  if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
    CUDA_CHECK(cudaMalloc(&d_gas_output_buffer, compacted_gas_size));

    // use handle as input and output
    OPTIX_CHECK(
        optixAccelCompact(optix_context, 0, gas_handle,
                          reinterpret_cast<CUdeviceptr>(d_gas_output_buffer),
                          compacted_gas_size, &gas_handle));

    CUDA_CHECK(cudaFree(d_buffer_temp_output_gas_and_compacted_size));
  } else {
    d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
  }
  CUDA_CHECK(cudaFree(d_vertices));
  return aabb;
}

void RTXDataHolder::setStream(const cudaStream_t &stream_in) {
  this->stream = stream_in;
}

OptixAabb RTXDataHolder::read_obj_mesh(const std::string &obj_filename,
                                       std::vector<float3> &vertices,
                                       std::vector<uint3> &triangles) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              obj_filename.c_str());
  OptixAabb aabb;
  aabb.minX = aabb.minY = aabb.minZ = std::numeric_limits<float>::max();
  aabb.maxX = aabb.maxY = aabb.maxZ = -std::numeric_limits<float>::max();

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return aabb;
  }

  for (size_t s = 0; s < shapes.size(); s++) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      int fv = shapes[s].mesh.num_face_vertices[f];

      auto vertexOffset = vertices.size();

      for (size_t v = 0; v < fv; v++) {
        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

        if (idx.vertex_index >= 0) {
          tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
          tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
          tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

          vertices.push_back(make_float3(vx, vy, vz));

          // Update aabb
          aabb.minX = std::min(aabb.minX, vx);
          aabb.minY = std::min(aabb.minY, vy);
          aabb.minZ = std::min(aabb.minZ, vz);

          aabb.maxX = std::max(aabb.maxX, vx);
          aabb.maxY = std::max(aabb.maxY, vy);
          aabb.maxZ = std::max(aabb.maxZ, vz);
        }
      }
      index_offset += fv;

      triangles.push_back(
          make_uint3(vertexOffset, vertexOffset + 1, vertexOffset + 2));
    }
  }
  return aabb;
}
