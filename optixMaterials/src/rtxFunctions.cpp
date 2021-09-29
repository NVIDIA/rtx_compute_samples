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

#include <algorithm>
#include <limits>

#include <optix.h>
#include <optix_function_table_definition.h>
 
#include "common/common.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"

void RTXDataHolder::initContext() {
  CUDA_CHECK(cudaFree(0)); // Initializes CUDA context
  OPTIX_CHECK(optixInit());
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &optixLogCallback;
  options.logCallbackLevel = 3; // This goes upto level 4
  OPTIX_CHECK(optixDeviceContextCreate(0, &options, &optix_context));
}

void RTXDataHolder::createModule(const std::string ptx_filename) {

  std::ifstream ptx_in(ptx_filename);
  if (!ptx_in) {
    std::cerr << "ERROR: readPTX() Failed to open file " << ptx_filename
              << std::endl;
    return;
  }

  std::string ptx = std::string((std::istreambuf_iterator<char>(ptx_in)),
                                std::istreambuf_iterator<char>());

  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.maxRegisterCount =
      OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
 
  pipeline_compile_options.usesMotionBlur = 0;
  pipeline_compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_compile_options.numPayloadValues = 2;
  pipeline_compile_options.numAttributeValues = 2;
  pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  OPTIX_CHECK(optixModuleCreateFromPTX(optix_context, &module_compile_options,
                                       &pipeline_compile_options, ptx.c_str(),
                                       ptx.size(), nullptr, nullptr, &module));
}

void RTXDataHolder::createProgramGroups() {

  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

  OptixProgramGroupDesc raygen_prog_group_desc = {}; // Ray Generation program
  raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_prog_group_desc.raygen.module = module;
  raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__prog";
  OPTIX_CHECK(optixProgramGroupCreate(optix_context, &raygen_prog_group_desc,
                                      1, // num program groups
                                      &program_group_options, nullptr, nullptr,
                                      &raygen_prog_group));

  OptixProgramGroupDesc miss_prog_group_desc = {}; // Miss program
  miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_prog_group_desc.miss.module = nullptr;
  miss_prog_group_desc.miss.entryFunctionName = nullptr;

  OPTIX_CHECK(optixProgramGroupCreate(optix_context, &miss_prog_group_desc,
                                      1, // num program groups
                                      &program_group_options, nullptr, nullptr,
                                      &miss_prog_group));

  OptixProgramGroupDesc hitgroup_prog_group_desc_plane = {}; // Hit group programs
  hitgroup_prog_group_desc_plane.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_prog_group_desc_plane.hitgroup.moduleCH = module;
  hitgroup_prog_group_desc_plane.hitgroup.entryFunctionNameCH = "__closesthit__prog_planes";
  hitgroup_prog_group_desc_plane.hitgroup.moduleAH = nullptr;
  hitgroup_prog_group_desc_plane.hitgroup.entryFunctionNameAH = nullptr;
 
  OptixProgramGroupDesc hitgroup_prog_group_desc_sphere = {}; // Hit group programs
  hitgroup_prog_group_desc_sphere.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_prog_group_desc_sphere.hitgroup.moduleCH = module;
  hitgroup_prog_group_desc_sphere.hitgroup.entryFunctionNameCH = "__closesthit__prog_sphere";
  hitgroup_prog_group_desc_sphere.hitgroup.moduleAH = nullptr;
  hitgroup_prog_group_desc_sphere.hitgroup.entryFunctionNameAH = nullptr;

  hitgroup_prog_groups.resize(2);
  OptixProgramGroupDesc hitgroup_prog_group_descs[2] ;
  hitgroup_prog_group_descs[PLANE] = hitgroup_prog_group_desc_plane;
  hitgroup_prog_group_descs[SPHERE] = hitgroup_prog_group_desc_sphere;
   
  OPTIX_CHECK(optixProgramGroupCreate(optix_context, hitgroup_prog_group_descs,
                                      2, // num program groups
                                      &program_group_options, nullptr, nullptr,
                                      &hitgroup_prog_groups[0]));
}

void RTXDataHolder::linkPipeline() {

  std::vector<OptixProgramGroup> program_groups = {raygen_prog_group, miss_prog_group,
                                        hitgroup_prog_groups[PLANE], hitgroup_prog_groups[SPHERE]};


  OptixPipelineLinkOptions pipeline_link_options = {};
  // This controls recursive depth of ray tracing. In this example we set to the
  // max bounce parameter
  pipeline_link_options.maxTraceDepth = 1;
  pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  OPTIX_CHECK(optixPipelineCreate(
      optix_context, &pipeline_compile_options, &pipeline_link_options,
      program_groups.data(),  program_groups.size(),
      nullptr, nullptr, &pipeline));
}

void RTXDataHolder::buildSBT() {
  void *raygenRecord;
  size_t raygenRecordSize = sizeof(RayGenSbtRecord);
  CUDA_CHECK(cudaMalloc((void **)&raygenRecord, raygenRecordSize));
  RayGenSbtRecord rgSBT;
  OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rgSBT));
  CUDA_CHECK(cudaMemcpy((void *)raygenRecord, &rgSBT, raygenRecordSize,
                        cudaMemcpyHostToDevice));

  void *missSbtRecord;
  size_t missSbtRecordSize = sizeof(MissSbtRecord);
  CUDA_CHECK(cudaMalloc((void **)&missSbtRecord, missSbtRecordSize));
  MissSbtRecord msSBT;
  OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &msSBT));
  CUDA_CHECK(cudaMemcpy(missSbtRecord, &msSBT, missSbtRecordSize,
                        cudaMemcpyHostToDevice));

  const size_t  hitgroupSbtRecordSize = 2;
  std::vector<HitGroupSbtRecord> hgSBT( hitgroupSbtRecordSize );
  void *hitgroupSbtRecord;
  CUDA_CHECK(cudaMalloc((void **)&hitgroupSbtRecord, hitgroupSbtRecordSize*sizeof(HitGroupSbtRecord)));
  OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[PLANE], &hgSBT[PLANE]));
  OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups[SPHERE], &hgSBT[SPHERE]));

  CUDA_CHECK(cudaMemcpy(hitgroupSbtRecord, &hgSBT[0], hitgroupSbtRecordSize*sizeof(HitGroupSbtRecord),
                        cudaMemcpyHostToDevice));

  sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygenRecord);
  sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(missSbtRecord);
  sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
  sbt.missRecordCount = 1;
  sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroupSbtRecord);
  sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
  sbt.hitgroupRecordCount = hitgroupSbtRecordSize;
}

OptixAabb
RTXDataHolder::buildAccelerationStructure( std::vector<std::vector<std::string>>& filesPerBuildInput) {

  const int nbuildInputs = filesPerBuildInput.size();
  std::vector<float3*> d_vertices(nbuildInputs);
  std::vector<void*>   d_triangles(nbuildInputs);

  std::vector<OptixBuildInput> triangle_input(nbuildInputs);
  std::vector<uint32_t> triangle_input_flags(nbuildInputs);

  OptixAabb aabb;
  aabb.minX = aabb.minY = aabb.minZ = std::numeric_limits<float>::max();
  aabb.maxX = aabb.maxY = aabb.maxZ = -std::numeric_limits<float>::max();


  unsigned long ntriangles = 0;
  for ( int buildInputId = 0; buildInputId<nbuildInputs; ++buildInputId)
  {
      std::vector<float3> vertices;
      std::vector<uint3> triangles;
       
    for ( int meshId = 0; meshId<filesPerBuildInput[buildInputId].size(); ++meshId)
    {
      read_obj_mesh(filesPerBuildInput[buildInputId][meshId], vertices, triangles, aabb);
      assert(vertices.size() > 0 && "Empty geometry, check .obj file location and content.");
    }
    ntriangles += triangles.size();
    const size_t vertices_size = sizeof(float3) * vertices.size();
    CUDA_CHECK(cudaMalloc(&d_vertices[buildInputId], vertices_size));
    CUDA_CHECK(cudaMemcpy(d_vertices[buildInputId], vertices.data(), vertices_size,
                          cudaMemcpyHostToDevice));

    const size_t tri_size = sizeof(uint3) * triangles.size();
  
    CUDA_CHECK(cudaMalloc(&d_triangles[buildInputId], tri_size));
    CUDA_CHECK(cudaMemcpy(d_triangles[buildInputId], triangles.data(), tri_size,
                          cudaMemcpyHostToDevice));

    triangle_input_flags[buildInputId] = OPTIX_GEOMETRY_FLAG_NONE;
    
    triangle_input[buildInputId].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input[buildInputId].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input[buildInputId].triangleArray.numVertices =  static_cast<unsigned int>(vertices.size());
    triangle_input[buildInputId].triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr *>(&d_vertices[buildInputId]);
    triangle_input[buildInputId].triangleArray.flags = &triangle_input_flags[buildInputId];
    triangle_input[buildInputId].triangleArray.numSbtRecords = 1;
    triangle_input[buildInputId].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input[buildInputId].triangleArray.numIndexTriplets =   static_cast<unsigned int>(triangles.size());
    triangle_input[buildInputId].triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(d_triangles[buildInputId]);

  }

  std::cout << "total number of  triangles "<< ntriangles << std::endl; 

  
  OptixAccelBuildOptions accel_options = {};
  // RANDOM VERTEX ACCESS flag allows access to triangle vertices while Ray
  // Tracing
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context, &accel_options,
                                           triangle_input.data(),
                                           nbuildInputs, // Number of build input
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
                              &accel_options,  triangle_input.data(),
                              nbuildInputs, // num build inputs
                              reinterpret_cast<CUdeviceptr>(d_temp_buffer_gas),
                              gas_buffer_sizes.tempSizeInBytes,
                              reinterpret_cast<CUdeviceptr>(
                                  d_buffer_temp_output_gas_and_compacted_size),
                              gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                              &emitProperty, // emitted property list
                              1              // num emitted properties
                              ));

  CUDA_CHECK(cudaFree(d_temp_buffer_gas));

   for ( int buildInputId = 0; buildInputId<nbuildInputs; ++buildInputId)
  {
     CUDA_CHECK(cudaFree(d_vertices[buildInputId]));
 CUDA_CHECK(cudaFree(d_triangles[buildInputId]));
  }

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
 
 return aabb;
}

void RTXDataHolder::setStream(const cudaStream_t &stream_in) {
  this->stream = stream_in;
}

RTXDataHolder::~RTXDataHolder() {
  OPTIX_CHECK(optixPipelineDestroy(pipeline));
  OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_groups[0]));
  OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_groups[1]));
  OPTIX_CHECK(optixModuleDestroy(module));
}

void RTXDataHolder::read_obj_mesh(const std::string &obj_filename,
                                       std::vector<float3> &vertices,
                                       std::vector<uint3> &triangles,
                                       OptixAabb& aabb ) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              obj_filename.c_str());

  if (!err.empty()) {
    std::cerr << err << std::endl;
  return;
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

}
