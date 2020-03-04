/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
uint32_t width = 2048u;
uint32_t height = 2048u;
uint32_t depth = 2048u;
float dx, dy, dz;

//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

Buffer get_output_buffer();
void destroy_context();
void create_context();
void load_mesh(const std::string &filename);
optix::Buffer dbufferfront, dbufferback, dbufferright, dbufferleft, dbuffertop,
    dbufferbottom;

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

optix::Buffer get_output_buffer() {
  return context["output_buffer"]->getBuffer();
}

void destroy_context() {
  if (context) {
    context->destroy();
    context = 0;
  }
}

void create_context() {

  // This is not necessary anymore. RTX is enabled automatically be OptiX 6.
  // if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(RTX), &RTX)
  // != RT_SUCCESS)
  //     printf("Error setting RTX mode. \n");
  // else
  //     printf("OptiX RTX execution mode is %s.\n", (RTX) ? "on" : "off");

  // Set up context
  context = Context::create();

  context->setRayTypeCount(2);
  context->setEntryPointCount(6);

  // context->setPrintEnabled(1);
  context["scene_epsilon"]->setFloat(1.e-4f);

  optix::Buffer buffer = context->createBuffer(
      RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
  context["output_buffer"]->setBuffer(buffer);

  dbufferfront =
      context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width, height);
  context["output_front"]->setBuffer(dbufferfront);
  dbufferback =
      context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width, height);
  context["output_back"]->setBuffer(dbufferback);
  dbuffertop =
      context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width, depth);
  context["output_top"]->setBuffer(dbuffertop);
  dbufferbottom =
      context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width, depth);
  context["output_bottom"]->setBuffer(dbufferbottom);
  dbufferleft =
      context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, depth, height);
  context["output_left"]->setBuffer(dbufferleft);
  dbufferright =
      context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, depth, height);
  context["output_right"]->setBuffer(dbufferright);

  // Ray generation program
  Program ray_gen_program0 =
      context->createProgramFromPTXFile("camera.ptx", "camera_front");
  context->setRayGenerationProgram(0, ray_gen_program0);

  Program ray_gen_program1 =
      context->createProgramFromPTXFile("camera.ptx", "camera_left");
  context->setRayGenerationProgram(2, ray_gen_program1);

  Program ray_gen_program2 =
      context->createProgramFromPTXFile("camera.ptx", "camera_back");
  context->setRayGenerationProgram(1, ray_gen_program2);

  Program ray_gen_program3 =
      context->createProgramFromPTXFile("camera.ptx", "camera_right");
  context->setRayGenerationProgram(3, ray_gen_program3);

  Program ray_gen_program4 =
      context->createProgramFromPTXFile("camera.ptx", "camera_top");
  context->setRayGenerationProgram(4, ray_gen_program4);

  Program ray_gen_program5 =
      context->createProgramFromPTXFile("camera.ptx", "camera_bottom");
  context->setRayGenerationProgram(5, ray_gen_program5);

  // Exception program
  Program exception_program =
      context->createProgramFromPTXFile("camera.ptx", "exception");
  context->setExceptionProgram(0, exception_program);
  context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

  // Miss program
  context->setMissProgram(
      0, context->createProgramFromPTXFile("camera.ptx", "miss"));
  context["bg_color"]->setFloat(0.85f, 0.30f, 0.34f);
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

  optix::Aabb aabb;
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

          // if (idx.normal_index >= 0)
          // {
          //     tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
          //     tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
          //     tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

          //     mesh.normals.push_back(VisRTX::Vec3f(nx, ny, nz));
          // }

          // if (idx.texcoord_index >= 0)
          // {
          //     tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index +
          //     0]; tinyobj::real_t ty = attrib.texcoords[2 *
          //     idx.texcoord_index + 1];

          //     mesh.texcoords.push_back(VisRTX::Vec2f(tx, ty));
          // }
        }
        index_offset += fv;

        triangles.push_back(optix::make_uint3(vertexOffset, vertexOffset + 1,
                                              vertexOffset + 2));
      }
    }
  }

  dx = (aabb.m_max.x - aabb.m_min.x) / width;
  dy = (aabb.m_max.y - aabb.m_min.y) / height;
  dz = (aabb.m_max.z - aabb.m_min.z) / depth;
  float3 farpoint = optix::make_float3(3, 3, 3);
  float traymax = 10;

  printf("minimum bbox (%f,%f,%f) \n", aabb.m_min.x, aabb.m_min.y,
         aabb.m_min.z);
  printf("maximum bbox (%f,%f,%f) \n", aabb.m_max.x, aabb.m_max.y,
         aabb.m_max.z);
  printf(" Delta values (%f,%f,%f)\n", dx, dy, dz);
  printf(" Far away launch point (%f,%f,%f) and tmax---> %f\n", farpoint.x,
         farpoint.y, farpoint.z, traymax);

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
  context["top_shadower"]->set(group);
  context["delta"]->setFloat(dx, dy, dz);
  context["minpoint"]->setFloat(aabb.m_min);
  context["far_away"]->setFloat(farpoint);
  context["tmax"]->setFloat(traymax);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------
void launch() {
  context->launch(0, width, height);
  context->launch(1, width, height);
  context->launch(2, depth, height);
  context->launch(3, depth, height);
  context->launch(4, width, depth);
  context->launch(5, width, depth);
}
int main(int argc, char **argv) {
  std::string testfile = "cow";
  std::string out_file = testfile + ".ppm";
  std::string mesh_file = testfile + ".obj";
  std::cout << mesh_file << std::endl;
  int usage_report_level = 0;

  create_context();

  auto start = std::chrono::system_clock::now();
  load_mesh(mesh_file);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_meshload = end - start;
  std::cout << "Mesh read+build time ---> " << elapsed_seconds_meshload.count()
            << " seconds " << std::endl;

  launch();
  int Ntimes = 10;
  context->validate();
  std::cout << "launching" << std::endl;

  start = std::chrono::system_clock::now();
  for (int i = 0; i < Ntimes; i++)
    launch();
  optix::cudaDeviceSynchronize();
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_compute =
      (end - start) / Ntimes;
  std::cout << "Projection compute time ---> "
            << elapsed_seconds_compute.count() << " seconds " << std::endl;

  // Convert to RGB
  optix::uchar4 *rgba =
      reinterpret_cast<optix::uchar4 *>(get_output_buffer()->map());
  std::vector<uint8_t> rgb(width * height * 3);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint32_t i = y * width + x;

      rgb[3 * i + 0] = rgba[i].x;
      rgb[3 * i + 1] = rgba[i].y;
      rgb[3 * i + 2] = rgba[i].z;
    }
  }
  get_output_buffer()->unmap();

  // Write PPM
  std::ofstream outFile;
  outFile.open(out_file.c_str(), std::ios::binary);

  outFile << "P6"
          << "\n"
          << width << " " << height << "\n"
          << "255\n";

  outFile.write((char *)rgb.data(), rgb.size());

  destroy_context();

  return 0;
}
