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

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "../common/tiny_obj_loader.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdint.h>
#include <vector>

using namespace optix;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------
const int RTX = true;
optix::Context context;
uint32_t n_particles = 1024;

//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

void destroy_context();
void create_context();
void init_particles();
void load_mesh(const std::string &filename);
optix::Buffer velocity_buffer, position_buffer;
optix::Buffer output_distance_buffer;
optix::Aabb aabb; // Bbox enclosing the entire geometry

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void destroy_context() {
  if (context) {
    context->destroy();
    context = 0;
  }
}

void create_context() {

  context = Context::create();
  context->setEntryPointCount(1);
  context->setRayTypeCount(1);

  // Print has to be enabled if, printf is to be used from Optix device code.
  // context->setPrintEnabled(1);

  // Create buffer to store output
  output_distance_buffer =
      context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, n_particles);
  context["output_distance"]->setBuffer(output_distance_buffer);

  // Create buffes to store particle velocity and position
  velocity_buffer =
      context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, n_particles);
  context["velocity"]->setBuffer(velocity_buffer);

  position_buffer =
      context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, n_particles);
  context["position"]->setBuffer(position_buffer);

  // Ray generation program
  Program ray_gen_program =
      context->createProgramFromPTXFile("camera.ptx", "generate_rays");
  context->setRayGenerationProgram(0, ray_gen_program);

  // Exception program
  Program exception_program =
      context->createProgramFromPTXFile("camera.ptx", "exception");
  context->setExceptionProgram(0, exception_program);

  // Miss program
  context->setMissProgram(
      0, context->createProgramFromPTXFile("camera.ptx", "miss"));
}

void load_mesh(const std::string &filename) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              filename.c_str());

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return;
  }

  aabb.m_min.x = aabb.m_min.y = aabb.m_min.z =
      std::numeric_limits<float>::max();
  aabb.m_max.x = aabb.m_max.y = aabb.m_max.z =
      -std::numeric_limits<float>::max();

  std::vector<optix::float3> vertices;
  std::vector<optix::uint3> triangles;

  for (size_t s = 0; s < shapes.size(); s++) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      int fv = shapes[s].mesh.num_face_vertices[f];

      if (fv == 3) {
        uint32_t vertexOffset = (uint32_t)vertices.size();

        for (size_t v = 0; v < fv; v++) {
          // access to vertex
          tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

          if (idx.vertex_index >= 0) {
            tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
            tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
            tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

            vertices.push_back(optix::make_float3(vx, vy, vz));

            // Update aabb
            aabb.m_min.x = std::min(aabb.m_min.x, vx);
            aabb.m_min.y = std::min(aabb.m_min.y, vy);
            aabb.m_min.z = std::min(aabb.m_min.z, vz);

            aabb.m_max.x = std::max(aabb.m_max.x, vx);
            aabb.m_max.y = std::max(aabb.m_max.y, vy);
            aabb.m_max.z = std::max(aabb.m_max.z, vz);
          }
        }
        index_offset += fv;

        triangles.push_back(optix::make_uint3(vertexOffset, vertexOffset + 1,
                                              vertexOffset + 2));
      }
    }
  }

  printf("minimum bbox (%f,%f,%f) \n", aabb.m_min.x, aabb.m_min.y,
         aabb.m_min.z);
  printf("maximum bbox (%f,%f,%f) \n", aabb.m_max.x, aabb.m_max.y,
         aabb.m_max.z);

  optix::Buffer vertexBuffer =
      context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vertices.size());
  void *p = vertexBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
  memcpy(p, vertices.data(), sizeof(optix::float3) * vertices.size());
  vertexBuffer->unmap();

  optix::Buffer triangleBuffer = context->createBuffer(
      RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, triangles.size());
  p = triangleBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
  memcpy(p, triangles.data(), sizeof(optix::uint3) * triangles.size());
  triangleBuffer->unmap();

  optix::GeometryTriangles geometry = context->createGeometryTriangles();
  geometry->setVertices(vertices.size(), vertexBuffer, 0, sizeof(optix::float3),
                        RT_FORMAT_FLOAT3);
  geometry->setTriangleIndices(triangleBuffer, 0, sizeof(optix::uint3),
                               RT_FORMAT_UNSIGNED_INT3);
  geometry->setPrimitiveCount(triangles.size());

  optix::Material material = context->createMaterial();
  material->setClosestHitProgram(
      0, context->createProgramFromPTXFile("camera.ptx", "closest_hit"));
  // material->setAnyHitProgram(0,
  // context->createProgramFromPTXFile("camera.ptx", "any_hit")); // Disable any
  // hit if not needed for increased performance

  optix::GeometryInstance instance = context->createGeometryInstance();
  instance->setGeometryTriangles(geometry);
  instance->setMaterialCount(1);
  instance->setMaterial(0, material);

  optix::GeometryGroup group = context->createGeometryGroup();
  group->addChild(instance);
  group->setAcceleration(context->createAcceleration("Trbvh", "bvh"));

  context["top_object"]->set(group);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------
void launch() { context->launch(0, n_particles); }

int main(int argc, char **argv) {
  std::string testfile = "cow";
  std::string mesh_file = testfile + ".obj";
  std::cout << mesh_file << std::endl;

  create_context();

  auto start = std::chrono::system_clock::now();
  load_mesh(mesh_file);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_meshload = end - start;
  std::cout << "Mesh read+build time ---> " << elapsed_seconds_meshload.count()
            << " seconds " << std::endl;

  init_particles();
  context->validate();

  std::cout << "launching" << std::endl;
  launch();
  cudaDeviceSynchronize();

  destroy_context();
  return 0;
}

float random_float_val(const float min, const float max) {
  assert(max > min);
  float random = ((float)rand()) / (float)RAND_MAX;
  float range = max - min;
  return (random * range) + min;
}

void init_particles() {
  std::vector<optix::float3> velocity, position;
  velocity.resize(n_particles);
  position.resize(n_particles);

  optix::float3 &min = aabb.m_min;
  optix::float3 &max = aabb.m_max;

  for (int i = 0; i < n_particles; i++) {
    position[i] = optix::make_float3(random_float_val(min.x, max.x),
                                     random_float_val(min.y, max.y),
                                     random_float_val(min.z, max.z));
    velocity[i] = optix::make_float3(random_float_val(0.0, 1.0),
                                     random_float_val(0.0, 1.0),
                                     random_float_val(0.0, 1.0));
  }
  void *p1 = velocity_buffer->map(0, RT_BUFFER_MAP_WRITE);
  memcpy(p1, velocity.data(), sizeof(optix::float3) * n_particles);
  velocity_buffer->unmap();

  void *p2 = position_buffer->map(0, RT_BUFFER_MAP_WRITE);
  memcpy(p2, position.data(), sizeof(optix::float3) * n_particles);
  position_buffer->unmap();

  return;
}
