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
  int hitcounter;
  int counter;
  bool hitflags[14];
};

rtDeclareVariable(float3, delta, , );
rtDeclareVariable(float3, minpoint, , );

rtBuffer<float, 3> output;
rtDeclareVariable(rtObject, top_object, , );

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim, rtLaunchDim, );
rtDeclareVariable(PerRayData, payload, rtPayload, );
rtDeclareVariable(optix::Ray, curr_ray, rtCurrentRay, );

rtBuffer<float3, 1> ray_direction;

RT_PROGRAM void ray_gen() {
  float xo = minpoint.x + delta.x * launch_index.x + (delta.x / 2);
  float yo = minpoint.y + delta.y * launch_index.y + (delta.y / 2);
  float zo = minpoint.z + delta.z * launch_index.z + (delta.z / 2);
  PerRayData prd;
  prd.hitcounter = 0;
  float3 ray_origin = make_float3(xo, yo, zo);
  optix::Ray ray;

  for (int i = 0; i < 14; i++) {
    prd.counter = i;
    prd.hitflags[i] = false;
    float3 dir = normalize(ray_direction[i]);
    float tmax = length(delta * ray_direction[i]);
    ray = optix::make_Ray(ray_origin, dir, 0, 0, tmax);
    rtTrace(top_object, ray, prd, RT_VISIBILITY_ALL, RT_RAY_FLAG_NONE);
  }
  output[launch_index] = (float)prd.hitcounter;
}

RT_PROGRAM void exception() { rtPrintExceptionDetails(); }

RT_PROGRAM void closest_hit() {}

RT_PROGRAM void any_hit() {
  payload.hitflags[payload.counter] = true;
  payload.hitcounter++;
  rtTerminateRay();
}

RT_PROGRAM void miss() {}
