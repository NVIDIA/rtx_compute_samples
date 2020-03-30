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
#include <optix_world.h>

using namespace optix;

struct PerRayData {
  float depth;
};

rtBuffer<float, 1> output_distance;
rtBuffer<float3, 1> velocity, position;

rtDeclareVariable(rtObject, top_object, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(PerRayData, payload, rtPayload, );
rtDeclareVariable(optix::Ray, curr_ray, rtCurrentRay, );

RT_PROGRAM void generate_rays() {
  float3 ray_origin =
      position[launch_index.x]; // Ray origin at the position of the particle
  float3 ray_direction = optix::normalize(
      velocity[launch_index.x]); // Ray direction along the velocity direction

  optix::Ray ray =
      optix::make_Ray(ray_origin, ray_direction, 0, 0, RT_DEFAULT_MAX);

  PerRayData prd;
  prd.depth = 0.0f;

  rtTrace(
      top_object, ray, prd, RT_VISIBILITY_ALL,
      RT_RAY_FLAG_DISABLE_ANYHIT); // or RT_RAY_FLAG_NONE if anyhit is required
  output_distance[launch_index.x] = prd.depth; // Save output to GPU memory
}

RT_PROGRAM void exception() { rtPrintExceptionDetails(); }

RT_PROGRAM void closest_hit() {
  payload.depth = curr_ray.tmax; // Record distance from particle positon to the
                                 // hit on the geometry
}

RT_PROGRAM void any_hit() {}

RT_PROGRAM void miss() {
  payload.depth = -1; // If nothing is hit, store -1
}
