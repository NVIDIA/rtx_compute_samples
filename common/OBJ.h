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

#include <cuda_runtime.h>

#include "common/tiny_obj_loader.h"

#include <string>
#include <vector>

class OBJ {
public:
  OBJ(std::string const &path, std::string const &mtl_path);

  size_t num_shapes(void) const { return shapes.size(); }
  size_t num_vertices(void) const {
    return vertex_attributes.vertices.size() / 3ul;
  }
  size_t num_triangles(void) const {
    size_t count = 0;
    for (auto const &cur_shape : shapes)
      count += cur_shape.mesh.num_face_vertices.size();
    return count;
  }

  bool has_vertices(void) const { return !vertex_attributes.vertices.empty(); }
  void get_vertices(std::vector<float3> &verts) const;

  bool has_normals(void) const { return !vertex_attributes.normals.empty(); }
  void get_normals(std::vector<float3> &normals) const;

  bool has_texture_coordinates(void) const {
    return !vertex_attributes.texcoords.empty();
  }
  void get_texture_coordinates(std::vector<float2> &tex_coords) const;

  bool has_colors(void) const { return !vertex_attributes.colors.empty(); }
  void get_colors(std::vector<float3> &colors) const;

  tinyobj::shape_t &get_shape(size_t shape_idx) { return shapes[shape_idx]; }
  tinyobj::shape_t const &get_shape(size_t shape_idx) const {
    return shapes[shape_idx];
  }

  // precondition: triangulated mesh
  void get_shape_vertex_indices(size_t shape_idx,
                                std::vector<uint3> &indices) const {
    auto const &mesh = get_shape(shape_idx).mesh;
    size_t const num_triangles = mesh.num_face_vertices.size();
    indices.resize(num_triangles);
    for (size_t i = 0; i < num_triangles; ++i) {
      indices[i] = make_uint3(mesh.indices[i * 3 + 0].vertex_index,
                              mesh.indices[i * 3 + 1].vertex_index,
                              mesh.indices[i * 3 + 2].vertex_index);
    }
  }

private:
  tinyobj::attrib_t vertex_attributes;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
};
