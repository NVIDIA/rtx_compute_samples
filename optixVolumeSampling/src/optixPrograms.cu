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

#include "params.hpp"
#include <optix.h>

extern "C" static __constant__ Params params;

extern "C" __global__ void __raygen__ray_march() {
  const uint3 launch_index = optixGetLaunchIndex();
  const float3 &delta = params.delta;
  const float3 &min_point = params.min_point;
  const float3 &max_point = params.max_point;

  float xo = min_point.x + delta.x * launch_index.x + (delta.x / 2);
  float yo = min_point.y + delta.y * launch_index.y + (delta.y / 2);
  float zo = -1 * min_point.z; // rays coming from the back plane to front
  float3 ray_origin = make_float3(xo, yo, zo);

  float3 ray_direction = make_float3(0.0, 0.0, 1.0);

  float tmin = 0.0f;
  float tmax = (max_point.z - min_point.z) + 100.0;
  float ray_time = 0.0f;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_NONE;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTIndex = 0;
  unsigned int payload = 0;
  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
             visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
             payload);

  unsigned int idx = launch_index.x + launch_index.y * params.width;
  params.output[idx] = __uint_as_float(payload);
}

extern "C" __global__ void __anyhit__ray_march() {
  // For every intersection, we update a scalar value
  float val = __uint_as_float(optixGetPayload_0());
  val += 0.2; // can be the scalar value associated with a triangle.
  optixSetPayload_0(__float_as_uint(val));
  optixIgnoreIntersection();
}

// extern "C" __global__ void __miss__ray_march() {}
// extern "C" __global__ void __closesthit__ray_march() {}

extern "C" __global__ void __raygen__ray_sample() {
  const uint3 launch_index = optixGetLaunchIndex();
  const float3 &delta = params.delta;
  const float3 &min_point = params.min_point;

  float xo = min_point.x + delta.x * launch_index.x + (delta.x / 2);
  float yo = min_point.y + delta.y * launch_index.y + (delta.y / 2);
  float zo = -1 * (min_point.z + delta.z * launch_index.z);
  float3 ray_origin = make_float3(xo, yo, zo);

  float3 ray_direction = make_float3(0.0, 0.0, 1.0);

  float tmin = 0.0f;
  float tmax = delta.z + 1;
  float ray_time = 0.0f;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTIndex = 0;
  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
             visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex);
}

// extern "C" __global__ void __anyhit__ray_sample() {}
// extern "C" __global__ void __miss__ray_sample() {}

extern "C" __global__ void __closesthit__ray_sample() {
  const uint3 launch_index = optixGetLaunchIndex();
  // For every closest hit, we update a scalar value in Global memory
  unsigned int idx = launch_index.x + launch_index.y * params.width;
  float *output = params.output;
  atomicAdd(output + idx,
            0.2f); // can be the scalar value associated with a triangle.
}