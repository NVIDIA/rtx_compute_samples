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
  bool issample;
  int counter;
  float value;
};

rtDeclareVariable(float3, delta, , );
rtDeclareVariable(float3, minpoint, , );
rtDeclareVariable(float, tmax, , );
rtDeclareVariable(float, tmax_delta, , );

rtBuffer<float, 1> output_raymarch;
rtBuffer<float, 1> output_raysample;
rtDeclareVariable(rtObject, top_object_raymarch, , );
rtDeclareVariable(rtObject, top_object_raysample, , );

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim, rtLaunchDim, );
rtDeclareVariable(PerRayData, payload, rtPayload, );
rtDeclareVariable(optix::Ray, curr_ray, rtCurrentRay, );

RT_PROGRAM void ray_gen_raymarch() {
  float xo = minpoint.x + delta.x * launch_index.x + (delta.x / 2);
  float yo = minpoint.y + delta.y * launch_index.y + (delta.y / 2);
  float zo = minpoint.z;
  PerRayData prd;
  prd.counter = 0;
  prd.value = 0;
  prd.issample = false;
  float3 ray_origin = make_float3(xo, yo, zo);
  optix::Ray ray = make_Ray(ray_origin, make_float3(0, 0, 1), 0, 0, tmax);
  rtTrace(top_object_raymarch, ray, prd, RT_VISIBILITY_ALL, RT_RAY_FLAG_NONE);
  // printf("ray id (%d,%d) and counter %d\n", launch_index.x, launch_index.y,
  //       prd.counter);
  output_raymarch[launch_index.x + launch_index.y * launch_dim.x] = prd.value;
}

RT_PROGRAM void exception() { rtPrintExceptionDetails(); }

RT_PROGRAM void closest_hit_raymarch() {}

RT_PROGRAM void any_hit_raymarch() {
  payload.counter++;
  payload.value += 0.2; // can be the scalar value associated with a triangle.
  rtIgnoreIntersection();
}

RT_PROGRAM void miss() {
  if (payload.issample) {
    atomicAdd(&output_raysample[launch_index.x + launch_index.y * launch_dim.x],
              0.2);
  }
}

RT_PROGRAM void ray_gen_raysample() {
  float xo = minpoint.x + delta.x * launch_index.x + (delta.x / 2);
  float yo = minpoint.y + delta.y * launch_index.y + (delta.y / 2);
  float zo = minpoint.z;
  PerRayData prd;
  prd.counter = 0;
  prd.value = 0;
  prd.issample = true;
  float3 ray_origin = make_float3(xo, yo, zo);
  float raylen = tmax_delta * (launch_index.z + 1);
  optix::Ray ray = make_Ray(ray_origin, make_float3(0, 0, 1), 0, 0, raylen);
  rtTrace(top_object_raysample, ray, prd, RT_VISIBILITY_ALL, RT_RAY_FLAG_NONE);
  // printf("ray id (%d,%d) and counter %d\n", launch_index.x, launch_index.y,
  //       prd.counter);
}

RT_PROGRAM void closest_hit_raysample() {}

RT_PROGRAM void any_hit_raysample() { rtIgnoreIntersection(); }

