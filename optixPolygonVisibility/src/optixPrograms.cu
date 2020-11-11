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

#include <cuda_runtime.h>
#include <optix.h>

#include "Params.h"
#include "common/vec_func.cuh"

extern "C" __constant__ static Params params;

extern "C" __global__ void __raygen__launch360(void) {
  // Calculate our unique particle index
  uint3 launch_index = optixGetLaunchIndex();
  unsigned int ray_id = launch_index.x;
  int const center_idx = (int)(ray_id / 360);
  float angle = (float)(ray_id % 360) * (M_PI / 180.0f);
  float3 ray_origin = params.ray_centers[center_idx];
  float3 ray_dir = make_float3(cosf(angle), sinf(angle), 0);

  // Initialize Payload
  // We use -1 as a sentinel to indicate miss events
  unsigned int p0 = __float_as_uint(-1.0f);

  optixTrace(params.handle, ray_origin,
             ray_dir,                       // Ray direction
             0.0f, 1000.0f,                 // [tmin, tmax]
             0.0f,                          // ray time
             OptixVisibilityMask(255),      // visibility mask (8bit)
             OPTIX_RAY_FLAG_DISABLE_ANYHIT, // ray flags
             0,                             // SBT offset
             0,                             // SBT stride
             0,                             // SBT miss index
             p0);

  float const t_value_to_hit = __uint_as_float(p0);
  // printf("Hit tmax for ray %d is %f\n", ray_id, t_value_to_hit);
  // Store the distance to hit point.
  params.hit_tmax[ray_id] = t_value_to_hit;
}

extern "C" __global__ void __closesthit__hit360(void) {
  float const tmax = optixGetRayTmax();
  optixSetPayload_0(__float_as_uint(tmax));
}
