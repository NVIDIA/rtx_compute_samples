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

extern "C" __global__ void __raygen__prog() {
  const uint3 launch_index = optixGetLaunchIndex(); 
 
  const float3 &min_corner = params.min_corner;
  const float3 &delta = params.delta;

  float xo = min_corner.x + delta.x * launch_index.x;
  float yo = min_corner.y + delta.y * launch_index.y;
  float zo = min_corner.z;

  // create a ray
  float3 ray_origin = make_float3(xo, yo, zo);
  float3 ray_direction = normalize(make_float3(0.0, 0.0, 1.0));

  float tmin = 0.0f;
  float tmax = delta.z + 2.0;
  float ray_time = 0.0f;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 0;
  unsigned int missSBTIndex = 0;

  unsigned int hit_type = MISS;
 
  //tracing ray
  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
             visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
             hit_type);

  //increment counter of according hit type (planes, sphere or miss)
  atomicAdd(&params.hitCounter[hit_type], 1); 

}

 extern "C" __global__ void __closesthit__prog_sphere() {
   
 #ifdef DEBUG_OPTIX
  float ray_tmax =  optixGetRayTmax();
  float3 ray_dir = optixGetWorldRayDirection();
  float3 ray_origin = optixGetWorldRayOrigin();
  float3 hit_point = ray_origin + ray_tmax * ray_dir;
  printf( " hit sphere at ( %f, %f, %f ) \n",hit_point.x,hit_point.y,hit_point.z );
#endif
 
 optixSetPayload_0(SPHERE);
}

// extern "C" __global__ void __miss__prog() {}
// extern "C" __global__ void __anyhit__prog() {}
