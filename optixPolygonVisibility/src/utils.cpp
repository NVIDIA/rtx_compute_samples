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
#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>

#include <optix.h>
#include <optix_stubs.h>

// Local Headers
#include "common/Timer.h"
#include "common/common.h"

/** @file
 * Utility functions for the Particle Collision sample
 */

/**
 * Takes a vector of float3 and computes its axis aligned bounding box which is
 * returned as OptixAabb.
 */
OptixAabb compute_bounds(std::vector<float3> const &vertices) {
  OptixAabb bbox;
  bbox.minX = bbox.minY = bbox.minZ = std::numeric_limits<float>::max();
  bbox.maxX = bbox.maxY = bbox.maxZ = std::numeric_limits<float>::lowest();

  for (float3 const &cur_vert : vertices) {
    bbox.minX = std::min(bbox.minX, cur_vert.x);
    bbox.minY = std::min(bbox.minY, cur_vert.y);
    bbox.minZ = std::min(bbox.minZ, cur_vert.z);

    bbox.maxX = std::max(bbox.maxX, cur_vert.x);
    bbox.maxY = std::max(bbox.maxY, cur_vert.y);
    bbox.maxZ = std::max(bbox.maxZ, cur_vert.z);
  }

  return bbox;
}

/**
 * Initializes the ray centers
 *
 * \p num_centers centers are initialized with random position
 * within the scenes bounding box. This bounding box is computed in
 * @ref load_mesh and the information is stored as Optix variables in the
 * context.
 *
 * @param num_centers Number of centers to create
 */
void initialize_raycenters(size_t num_centers, OptixAabb const &bbox,
                           float3 **d_ray_centers) {
  // Set some handy aliases
  using container_t = std::vector<float3>;
  using dist_t = std::uniform_real_distribution<float>;
  using rng_t = std::mt19937;

  // Retrieve bounding box information from the context
  // this has been stored in @ref load_mesh
  float3 const &bbox_min = make_float3(bbox.minX, bbox.minY, bbox.minZ);
  float3 const &bbox_max = make_float3(bbox.maxX, bbox.maxY, bbox.maxZ);

  // Initialize the random number generator
  unsigned int const seed = 1234;
  rng_t rng(seed);

  // Lambda function that initializes a vector of float3 using the provided
  // distributions.
  auto init_rnd_vec = [num_centers, &rng](container_t &v,
                                          dist_t distX = dist_t(-1.0f, +1.0f),
                                          dist_t distY = dist_t(-1.0f, +1.0f),
                                          dist_t distZ = dist_t(-1.0f, +1.0f)) {
    v.reserve(num_centers);

    for (size_t i = 0; i < num_centers; ++i) {
      float x = distX(rng), y = distY(rng);
      v.push_back(make_float3(x, y, 0.0));
    }
  };

  // Lambda function that creates a new input buffer and uploads the vector
  // data to it. This makes the data available from within the ray generation
  // program.
  auto upload_to_buffer = [](container_t const &v) {
    void *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, sizeof(container_t::value_type) * v.size()));
    CUDA_CHECK(cudaMemcpy(d_buf, v.data(),
                          sizeof(container_t::value_type) * v.size(),
                          cudaMemcpyHostToDevice));
    return d_buf;
  };

  // Setup ray centers
  container_t ray_centers;

  // Use the bounding box information for the range of the distributions. This
  // creates random positions within the bounding box of our scene.
  init_rnd_vec(ray_centers, dist_t(bbox_min.x, bbox_max.x),
               dist_t(bbox_min.y, bbox_max.y), dist_t(bbox_min.z, bbox_max.z));

  // upload to optix
  *d_ray_centers = (float3 *)upload_to_buffer(ray_centers);

  // The data is uploaded, so we don't need it on the host side anymore
  ray_centers.clear();
}

template <typename T, typename K>
void build_2d_geometry(const float3 min, const float3 max,
                       const size_t num_polygons, T &vertices, K &triangles) {
  vertices.reserve(3 * num_polygons);
  triangles.reserve(3 * num_polygons);
  using dist_t = std::uniform_real_distribution<float>;
  using rng_t = std::mt19937;
  float radius = 1.0f;
  // Initialize the random number generator
  unsigned int const seed = 86411;
  rng_t rng(seed);
  dist_t distX = dist_t(min.x, max.x);
  dist_t distY = dist_t(min.y, max.y);
  dist_t distDeg = dist_t(0, 2 * M_PI);
  for (int i = 0; i < num_polygons; i++) {
    float p2deg = distDeg(rng), p3deg = distDeg(rng);
    float2 p1 = make_float2(distX(rng), distY(rng));
    float2 p2 =
        make_float2(p1.x + radius * cosf(p2deg), p1.y + radius * cosf(p2deg));
    float2 p3 =
        make_float2(p1.x + radius * cosf(p3deg), p1.y + radius * cosf(p3deg));
    vertices.push_back(p1);
    vertices.push_back(p2);
    vertices.push_back(p3);
    triangles.push_back(make_uint3(3 * i, 3 * i + 1, 3 * i + 2));
  }
}

template <typename L, typename K>
void raise_edge_3d(const float2 p1, const float2 p2, L &vertices3d,
                   K &triangles3d, const float delta, OptixAabb &aabb) {
  float3 p1_up = make_float3(p1.x, p1.y, delta);
  float3 p1_dw = make_float3(p1.x, p1.y, -1 * delta);
  float3 p2_up = make_float3(p2.x, p2.y, delta);
  float3 p2_dw = make_float3(p2.x, p2.y, -1 * delta);
  int curr_sz = vertices3d.size();
  vertices3d.push_back(p1_up);
  vertices3d.push_back(p1_dw);
  vertices3d.push_back(p2_dw);
  vertices3d.push_back(p2_up);
  triangles3d.push_back(make_uint3(curr_sz, curr_sz + 1, curr_sz + 2));
  triangles3d.push_back(make_uint3(curr_sz, curr_sz + 3, curr_sz + 2));

  // Update aabb
  for (int i = curr_sz; i < 4; i++) {
    float3 pnt = vertices3d[i];
    aabb.minX = std::min(aabb.minX, pnt.x);
    aabb.minY = std::min(aabb.minY, pnt.y);
    aabb.minZ = std::min(aabb.minZ, pnt.z);

    aabb.maxX = std::max(aabb.maxX, pnt.x);
    aabb.maxY = std::max(aabb.maxY, pnt.y);
    aabb.maxZ = std::max(aabb.maxZ, pnt.z);
  }
}

template <typename T, typename K, typename L>
OptixAabb elevate_2d_to_3d(const size_t num_polygons, T &vertices, K &triangles,
                           L &vertices3d, K &triangles3d, const float delta) {
  OptixAabb aabb;
  aabb.minX = aabb.minY = aabb.minZ = std::numeric_limits<float>::max();
  aabb.maxX = aabb.maxY = aabb.maxZ = -std::numeric_limits<float>::max();

  for (int i = 0; i < num_polygons; i++) {
    uint3 id = triangles[i];
    float2 p1 = vertices[id.x];
    float2 p2 = vertices[id.y];
    float2 p3 = vertices[id.z];
    raise_edge_3d(p1, p2, vertices3d, triangles3d, delta, aabb);
    raise_edge_3d(p1, p3, vertices3d, triangles3d, delta, aabb);
    raise_edge_3d(p2, p3, vertices3d, triangles3d, delta, aabb);
  }

  return aabb;
}

void print_usage(std::string const &cmd) {
  std::cout << "Usage: \n"
               "\t"
            << cmd << " [num_polygons] [num_points] [output.csv]\n\n";
}

// Dumps hit_points data into a csv file for simple visualization
void write_csv(std::string const &path, float *hit_tmax, size_t N_rays) {
  std::cout << "Writing output to '" + path + "'\n";
  std::ofstream file(path);
  if (!file)
    throw std::runtime_error("Could not open file for writing: '" + path + "'");

  // Write out all particles
  for (size_t i = 0; i < N_rays; ++i) {
    file << std::setprecision(16) << hit_tmax[i] << '\n';
  }
}