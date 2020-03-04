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

#include "helpers.h"
#include <optix_world.h>

using namespace optix;

struct PerRayData_radiance {
  float3 result;
  float depth;
};

rtDeclareVariable(float3, delta, , );
rtDeclareVariable(float3, minpoint, , );
rtDeclareVariable(float3, far_away, , );
rtDeclareVariable(float, tmax, , );

rtBuffer<uchar4, 2> output_buffer;
rtBuffer<float, 2> output_front, output_back, output_left, output_right,
    output_top, output_bottom;
rtDeclareVariable(rtObject, top_object, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(PerRayData_radiance, payload, rtPayload, );
rtDeclareVariable(optix::Ray, curr_ray, rtCurrentRay, );

rtBuffer<float3, 1> rayorigin;

RT_PROGRAM void camera_front() {

  float3 ray_origin =
      make_float3(minpoint.x + delta.x * launch_index.x,
                  minpoint.y + delta.y * launch_index.y, far_away.z);
  float3 ray_direction = make_float3(0.0, 0.0, -1.0);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, 0, tmax);

  PerRayData_radiance prd;
  prd.result = make_float3(0.0f, 0.0f, 0.0f);
  prd.depth = 0;

  rtTrace(
      top_object, ray, prd, RT_VISIBILITY_ALL,
      RT_RAY_FLAG_DISABLE_ANYHIT); // or RT_RAY_FLAG_NONE if anyhit is required
  output_buffer[launch_index] = make_color(prd.result);
  output_front[launch_index] = prd.depth;
}
RT_PROGRAM void camera_back() {
  float3 ray_origin =
      make_float3(minpoint.x + delta.x * launch_index.x,
                  minpoint.y + delta.y * launch_index.y, -1 * far_away.z);
  float3 ray_direction = make_float3(0.0, 0.0, 1.0);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, 0, tmax);

  PerRayData_radiance prd;
  prd.depth = 0;

  rtTrace(
      top_object, ray, prd, RT_VISIBILITY_ALL,
      RT_RAY_FLAG_DISABLE_ANYHIT); // or RT_RAY_FLAG_NONE if anyhit is required
  output_back[launch_index] = prd.depth;
}

RT_PROGRAM void camera_left() {
  float3 ray_origin =
      make_float3(-1 * far_away.x, minpoint.y + delta.y * launch_index.x,
                  minpoint.z + delta.z * launch_index.y);
  float3 ray_direction = make_float3(1.0, 0.0, 0.0);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, 0, tmax);

  PerRayData_radiance prd;
  prd.depth = 0;

  rtTrace(
      top_object, ray, prd, RT_VISIBILITY_ALL,
      RT_RAY_FLAG_DISABLE_ANYHIT); // or RT_RAY_FLAG_NONE if anyhit is required
  output_left[launch_index] = prd.depth;
}

RT_PROGRAM void camera_right() {
  float3 ray_origin =
      make_float3(far_away.x, minpoint.y + delta.y * launch_index.x,
                  minpoint.z + delta.z * launch_index.y);
  float3 ray_direction = make_float3(-1.0, 0.0, 0.0);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, 0, tmax);

  PerRayData_radiance prd;
  prd.depth = 0;

  rtTrace(
      top_object, ray, prd, RT_VISIBILITY_ALL,
      RT_RAY_FLAG_DISABLE_ANYHIT); // or RT_RAY_FLAG_NONE if anyhit is required
  output_right[launch_index] = prd.depth;
}

RT_PROGRAM void camera_top() {
  float3 ray_origin =
      make_float3(minpoint.x + delta.x * launch_index.x, far_away.y,
                  minpoint.z + delta.z * launch_index.y);
  float3 ray_direction = make_float3(0.0, -1.0, 0.0);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, 0, tmax);

  PerRayData_radiance prd;
  prd.depth = 0;

  rtTrace(
      top_object, ray, prd, RT_VISIBILITY_ALL,
      RT_RAY_FLAG_DISABLE_ANYHIT); // or RT_RAY_FLAG_NONE if anyhit is required
  output_top[launch_index] = prd.depth;
}

RT_PROGRAM void camera_bottom() {
  float3 ray_origin =
      make_float3(minpoint.x + delta.x * launch_index.x, -1 * far_away.y,
                  minpoint.z + delta.z * launch_index.y);
  float3 ray_direction = make_float3(0.0, 1.0, 0.0);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, 0, tmax);

  PerRayData_radiance prd;
  prd.depth = 0;

  rtTrace(
      top_object, ray, prd, RT_VISIBILITY_ALL,
      RT_RAY_FLAG_DISABLE_ANYHIT); // or RT_RAY_FLAG_NONE if anyhit is required
  output_bottom[launch_index] = prd.depth;
}

RT_PROGRAM void exception() { rtPrintExceptionDetails(); }

RT_PROGRAM void closest_hit() {
  payload.result = make_float3(1, 1, 1);
  payload.depth = 1.0;
}

RT_PROGRAM void any_hit() {}

RT_PROGRAM void miss() {}
