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

#include "OBJ.h"

#include <cassert>
#include <exception>
#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"

OBJ::OBJ(std::string const &path, std::string const &mtl_path) {
  // Load OBJ
  // This by default automatically triangulates the mesh
  std::string warn, err;
  tinyobj::LoadObj(&vertex_attributes, &shapes, &materials, &warn, &err,
                   path.c_str(), mtl_path.c_str());

  // Check if anything went wrong
  if (!warn.empty()) {
    std::cout << "OBJ Loading Warnings: \n\t" << warn << "\n";
  }
  if (!err.empty()) {
    throw std::runtime_error("Error Loading mesh: \n\t" + err + "\n");
  }
}

void OBJ::get_vertices(std::vector<float3> &verts) const {
  assert(has_vertices());
  size_t const N = num_vertices();
  verts.resize(N);
  for (size_t i = 0; i < N; ++i)
    verts[i] = make_float3(vertex_attributes.vertices[i * 3 + 0],
                           vertex_attributes.vertices[i * 3 + 1],
                           vertex_attributes.vertices[i * 3 + 2]);
}

void OBJ::get_normals(std::vector<float3> &normals) const {
  assert(has_normals());
  size_t const N = num_vertices();
  normals.resize(N);
  for (size_t i = 0; i < N; ++i)
    normals[i] = make_float3(vertex_attributes.normals[i * 3 + 0],
                             vertex_attributes.normals[i * 3 + 1],
                             vertex_attributes.normals[i * 3 + 2]);
}

void OBJ::get_texture_coordinates(std::vector<float2> &tex_coords) const {
  assert(has_texture_coordinates());
  size_t const N = num_vertices();
  tex_coords.resize(N);
  for (size_t i = 0; i < N; ++i)
    tex_coords[i] = make_float2(vertex_attributes.texcoords[i * 2 + 0],
                                vertex_attributes.texcoords[i * 2 + 1]);
}

void OBJ::get_colors(std::vector<float3> &colors) const {
  assert(has_colors());
  size_t const N = num_vertices();
  colors.resize(N);
  for (size_t i = 0; i < N; ++i)
    colors[i] = make_float3(vertex_attributes.colors[i * 3 + 0],
                            vertex_attributes.colors[i * 3 + 1],
                            vertex_attributes.colors[i * 3 + 2]);
}
