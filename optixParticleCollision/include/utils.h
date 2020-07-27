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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>

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
 * Initializes the positions and velocities of the particles
 *
 * \p num_particles Particles are initialized with random position and
 * velocities within the scenes bounding box. This bounding box is computed in
 * @ref load_mesh and the information is stored as Optix variables in the
 * context.
 *
 * @param num_particles Number of particles to create
 */
void initialize_particles(size_t num_particles, OptixAabb const &bbox,
                          float3 **d_particle_positions,
                          float3 **d_particle_velocities) {
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
  auto init_rnd_vec =
      [num_particles, &rng](container_t &v, dist_t distX = dist_t(-1.0f, +1.0f),
                            dist_t distY = dist_t(-1.0f, +1.0f),
                            dist_t distZ = dist_t(-1.0f, +1.0f)) {
        v.reserve(num_particles);

        for (size_t i = 0; i < num_particles; ++i) {
          float x = distX(rng), y = distY(rng), z = distZ(rng);
          v.push_back(make_float3(x, y, z));
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

  // Setup particle positions
  container_t particle_positions;

  // Use the bounding box information for the range of the distributions. This
  // creates random positions within the bounding box of our scene.
  init_rnd_vec(particle_positions, dist_t(bbox_min.x, bbox_max.x),
               dist_t(bbox_min.y, bbox_max.y), dist_t(bbox_min.z, bbox_max.z));

  // upload to optix
  *d_particle_positions = (float3 *)upload_to_buffer(particle_positions);

  // The data is uploaded, so we don't need it on the host side anymore
  particle_positions.clear();

  // Setup particle velocities
  container_t particle_velocities;

  // the distributions generate values in the [-1,1] range which is fine for
  // our velocities
  init_rnd_vec(particle_velocities);

  // upload to optix
  *d_particle_velocities = (float3 *)upload_to_buffer(particle_velocities);
}
