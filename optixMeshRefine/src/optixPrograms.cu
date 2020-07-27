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

#include "common/vec_func.cuh"
#include "params.hpp"
#include <optix.h>

extern "C" static __constant__ Params params;

extern "C" __global__ void __raygen__meshrefine() {
  const uint3 launch_index = optixGetLaunchIndex();
  const float3 &delta = params.delta;
  const float3 &min_point = params.min_point;
  unsigned int thread_idx = launch_index.x + launch_index.y * params.width +
                            launch_index.z * params.width * params.height;

  float xo = min_point.x + delta.x * launch_index.x + (delta.x / 2);
  float yo = min_point.y + delta.y * launch_index.y + (delta.y / 2);
  float zo = min_point.z + delta.z * launch_index.z + (delta.z / 2);
  float3 ray_origin = make_float3(xo, yo, zo);

  float tmin = 0.0f;
  float ray_time = 0.0f;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_NONE;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTIndex = 0;

  for (int i = 0; i < 14; i++) {
    unsigned int payload = 0;
    float3 ray_dir = params.ray_direction[i];
    float3 ray_direction = normalize(ray_dir);
    float tmax = length(delta * ray_dir) + 1;
    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
               visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
               payload);
    if (payload == 1) {
      params.output[thread_idx] = payload;
      break; // Break if atleast one ray hits.
    }
  }
}

// extern "C" __global__ void __miss__meshrefine() {}

extern "C" __global__ void __anyhit__meshrefine() {
  // If the geometry is hit, we set a flag. This flag can be then checked to
  // identify, which cells in the mesh needs refinement
  optixSetPayload_0(1);
  optixTerminateRay();
}

// extern "C" __global__ void __closesthit__meshrefine() {}
