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

extern "C" __constant__ static PipelineParams params;

extern "C" __global__ void __raygen__camera(void) {
  // Calculate our unique particle index
  uint3 launch_index = optixGetLaunchIndex();
  unsigned int const particle_idx = launch_index.x;

  float3 const cur_position = params.particle_positions[particle_idx];
  float3 const cur_velocity = params.particle_velocities[particle_idx];

  // Initialize Payload
  // We use -1 as a sentinel to indicate miss events
  unsigned int p0 = __float_as_uint(-1.0f);
  optixTrace(params.handle, cur_position, cur_velocity, 0.0f,
             1e30f,                    // [tmin, tmax]
             0.0f,                     // ray time
             OptixVisibilityMask(255), // visibility mask (8bit)
             OPTIX_RAY_FLAG_NONE,      // ray flags
             0,                        // SBT offset
             0,                        // SBT stride
             0,                        // SBT miss index
             p0);
  float const t_value_to_hit = __uint_as_float(p0);
  if (t_value_to_hit >= 0.0f) // ie. there was a hit
  {
    // for simplicity, we just move the particle to where it hit, ignoring
    // potential timestepping in a real particle simulation
    params.particle_positions[particle_idx] =
        cur_position + t_value_to_hit * cur_velocity;
  } else // ie. there was a miss
  {
    // for simplicity, we just move the particles that don't hit anything
    // to the origin
    params.particle_positions[particle_idx] = make_float3(0.0f, 0.0f, 0.0f);
  }
}

extern "C" __global__ void __closesthit__(void) {
  // This is NOT necessarilay the same as distance!
  // This is the parameterization of the ray and thus depends on the scaling
  // of the direction vector which, in our case, is NOT normalized.
  float const tmax = optixGetRayTmax();

  // Set our payload to the tmax value
  optixSetPayload_0(__float_as_uint(tmax));
}
