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

#include "optix.h"
#include <chrono>
#include <limits>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <string.h>
#include <vector>

using namespace optix;
void read_volume(std::vector<float3> &vertices, std::vector<uint3> &triangles,
                 std::string filename);

const int RTX = true;
Context context;
Buffer outputbuff_raymarch, outputbuff_raysample;
uint32_t width = 8u;
uint32_t height = 8u;
uint32_t depth = 120u; // 120 to make sure both test produce same results, for
                       // the wavelet volume

void print_output_buffer(Buffer &outbuff) {
  if (width <= 8 && height <= 8) {
    std::vector<float> outdata;
    outdata.resize(width * height);
    void *p = outbuff->map(0, RT_BUFFER_MAP_READ);
    memcpy(outdata.data(), p, sizeof(float) * width * height);
    outbuff->unmap();

    printf("\n");
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        printf("%0.2f  ", outdata[j + i * width]);
      }
      printf("\n");
    }
  } else {
    printf("Not printing output for width / height more than 8\n\n");
  }
}
void create_context() {

  context = Context::create();
  context->setRayTypeCount(1);
  context->setEntryPointCount(2);

  context->setPrintEnabled(1);
  outputbuff_raymarch =
      context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width * height);
  context["output_raymarch"]->setBuffer(outputbuff_raymarch);

  outputbuff_raysample =
      context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width * height);
  context["output_raysample"]->setBuffer(outputbuff_raysample);

  // Ray generation program
  Program ray_gen_program_raymarch =
      context->createProgramFromPTXFile("camera.ptx", "ray_gen_raymarch");
  context->setRayGenerationProgram(0, ray_gen_program_raymarch);

  Program ray_gen_program_raysample =
      context->createProgramFromPTXFile("camera.ptx", "ray_gen_raysample");
  context->setRayGenerationProgram(1, ray_gen_program_raysample);

  // Exception program
  Program exception_program =
      context->createProgramFromPTXFile("camera.ptx", "exception");
  context->setExceptionProgram(0, exception_program);

  // Miss program
  context->setMissProgram(
      0, context->createProgramFromPTXFile("camera.ptx", "miss"));
}

void build_mesh(std::vector<float3> &vertices, std::vector<uint3> &triangles) {

  Aabb aabb;
  aabb.m_min.x = aabb.m_min.y = aabb.m_min.z =
      std::numeric_limits<float>::max();
  aabb.m_max.x = aabb.m_max.y = aabb.m_max.z =
      -std::numeric_limits<float>::max();
  for (int i = 0; i < vertices.size(); i++) {
    float3 &vertex = vertices[i];
    if (aabb.m_min.x > vertex.x)
      aabb.m_min.x = vertex.x;
    if (aabb.m_min.y > vertex.z)
      aabb.m_min.y = vertex.y;
    if (aabb.m_min.z > vertex.z)
      aabb.m_min.z = vertex.z;
    if (aabb.m_max.x < vertex.x)
      aabb.m_max.x = vertex.x;
    if (aabb.m_max.y < vertex.y)
      aabb.m_max.y = vertex.y;
    if (aabb.m_max.z < vertex.z)
      aabb.m_max.z = vertex.z;
  }

  printf("minimum bbox (%f,%f,%f) \n", aabb.m_min.x, aabb.m_min.y,
         aabb.m_min.z);
  printf("maximum bbox (%f,%f,%f) \n", aabb.m_max.x, aabb.m_max.y,
         aabb.m_max.z);

  Buffer vertexBuffer =
      context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vertices.size());
  void *p = vertexBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
  memcpy(p, vertices.data(), sizeof(float3) * vertices.size());
  vertexBuffer->unmap();

  Buffer triangleBuffer = context->createBuffer(
      RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, triangles.size());
  p = triangleBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
  memcpy(p, triangles.data(), sizeof(uint3) * triangles.size());
  triangleBuffer->unmap();

  GeometryTriangles geometry = context->createGeometryTriangles();
  geometry->setVertices(vertices.size(), vertexBuffer, 0, sizeof(float3),
                        RT_FORMAT_FLOAT3);
  geometry->setTriangleIndices(triangleBuffer, 0, sizeof(uint3),
                               RT_FORMAT_UNSIGNED_INT3);
  geometry->setPrimitiveCount(triangles.size());

  Material material_raymarch = context->createMaterial();
  material_raymarch->setClosestHitProgram(
      0,
      context->createProgramFromPTXFile("camera.ptx", "closest_hit_raymarch"));
  material_raymarch->setAnyHitProgram(
      0, context->createProgramFromPTXFile("camera.ptx", "any_hit_raymarch"));

  Material material_raysample = context->createMaterial();
  material_raysample->setClosestHitProgram(
      0,
      context->createProgramFromPTXFile("camera.ptx", "closest_hit_raysample"));
  material_raysample->setAnyHitProgram(
      0, context->createProgramFromPTXFile("camera.ptx", "any_hit_raysample"));

  GeometryInstance instance_raymarch = context->createGeometryInstance();
  instance_raymarch->setGeometryTriangles(geometry);
  instance_raymarch->setMaterialCount(1);
  instance_raymarch->setMaterial(0, material_raymarch);

  GeometryInstance instance_raysample = context->createGeometryInstance();
  instance_raysample->setGeometryTriangles(geometry);
  instance_raysample->setMaterialCount(1);
  instance_raysample->setMaterial(0, material_raysample);

  GeometryGroup group_raymarch = context->createGeometryGroup();
  group_raymarch->addChild(instance_raymarch);
  group_raymarch->setAcceleration(context->createAcceleration("Trbvh", "bvh"));

  GeometryGroup group_raysample = context->createGeometryGroup();
  group_raysample->addChild(instance_raysample);
  group_raysample->setAcceleration(context->createAcceleration("Trbvh", "bvh"));

  // adding a bit od delta to the full bounding box
  aabb.m_min.x -= 1;
  aabb.m_min.y -= 1;
  aabb.m_min.z -= 1;
  aabb.m_max.x += 1;
  aabb.m_max.y += 1;
  aabb.m_max.z += 1;

  context["top_object_raymarch"]->set(group_raymarch);
  context["top_object_raysample"]->set(group_raysample);
  context["minpoint"]->setFloat(aabb.m_min.x, aabb.m_min.y, aabb.m_min.z);
  context["delta"]->setFloat((aabb.m_max.x - aabb.m_min.x) / width,
                             (aabb.m_max.y - aabb.m_min.y) / height,
                             (aabb.m_max.z - aabb.m_min.z) / depth);
  float tmax_len = aabb.m_max.z - aabb.m_min.z;

  context["tmax"]->setFloat(tmax_len + 10);
  context["tmax_delta"]->setFloat((tmax_len / depth));
}

void destroy_context() {
  if (context) {
    context->destroy();
    context = 0;
  }
}
std::chrono::duration<double> launch_raymarch() {
  context->validate();
  int Ntimes = 1;
  std::cout << "launching" << std::endl;

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < Ntimes; i++)
    context->launch(0, width, height, 1);

  cudaDeviceSynchronize();
  auto end = std::chrono::system_clock::now();
  return ((end - start) / Ntimes);
}

std::chrono::duration<double> launch_raysample() {
  context->validate();
  int Ntimes = 1;
  std::cout << "launching" << std::endl;

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < Ntimes; i++)
    context->launch(1, width, height, depth);

  cudaDeviceSynchronize();
  auto end = std::chrono::system_clock::now();
  return ((end - start) / Ntimes);
}

int main() {
  std::vector<float3> vertices;
  std::vector<uint3> triangles;
  read_volume(vertices, triangles, std::string("wavelet.txt"));
  std::cout << "No of vertices " << vertices.size() << "\n";
  std::cout << "No of triangles " << triangles.size() << "\n";
  create_context();
  build_mesh(vertices, triangles);
  std::chrono::duration<double> elapsed_seconds_compute;

  elapsed_seconds_compute = launch_raymarch();
  std::cout << "Volume sampling raymarch compute time ---> "
            << elapsed_seconds_compute.count() << " seconds " << std::endl;

  elapsed_seconds_compute = launch_raysample();
  std::cout << "Volume sampling raysample compute time ---> "
            << elapsed_seconds_compute.count() << " seconds " << std::endl;
  print_output_buffer(outputbuff_raymarch);
  print_output_buffer(outputbuff_raysample);
  destroy_context();
}
