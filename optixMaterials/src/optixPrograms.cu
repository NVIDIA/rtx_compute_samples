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

struct Payload {
  unsigned int ray_id; // unique id of the ray
  float tpath;         // total lenth of the path with multiple bounces
};

extern "C" __global__ void __raygen__prog() {
  const uint3 launch_index = optixGetLaunchIndex();
  const uint3 launch_dim = optixGetLaunchDimensions();

  const float3 &min_corner = params.min_corner;
  const float3 &delta = params.delta;

  float xo = min_corner.x + delta.x * launch_index.x;
  float yo = min_corner.y + delta.y * launch_index.y;
  float zo = min_corner.z;

  // setting the per ray data (payload)
  Payload pld;
  pld.tpath = 0.0f;
  pld.ray_id = launch_index.x + launch_dim.x * launch_index.y;

  // create a ray
  float3 ray_origin = make_float3(xo, yo, zo);
  float3 ray_direction = normalize(make_float3(0.0, 0.0, 1.0));

  float tmin = 0.0f;
  float tmax = delta.z + 100.0;
  float ray_time = 0.0f;
  OptixVisibilityMask visibilityMask = 255;
  unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  unsigned int SBToffset = 0;
  unsigned int SBTstride = 1;
  unsigned int missSBTIndex = 0;

  // Extract Payload as unsigned int
  unsigned int ray_id_payload = pld.ray_id;
  unsigned int tpath_payload = __float_as_uint(pld.tpath);

  optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
             visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
             ray_id_payload, tpath_payload);

  // Store back paylaod to the Payload struct
  pld.ray_id = ray_id_payload;
  pld.tpath = __uint_as_float(tpath_payload);

  // store the result and  number of bounces back to the global buffers
  params.tpath[pld.ray_id] = pld.tpath;
}

extern "C" __global__ void __closesthit__prog_planes() {
  unsigned int tri_id = optixGetPrimitiveIndex();
  // We defined out geometry as a triangle geometry. In this case the
  // We add the t value of the intersection
 

  unsigned int ray_id_payload = optixGetPayload_0();

  float ray_tmax =  optixGetRayTmax();
 float3 ray_dir = optixGetWorldRayDirection();
  float3 ray_origin = optixGetWorldRayOrigin();
  float3 hit_point = ray_origin + ray_tmax * ray_dir;

  printf( " hit planes at ( %f, %f, %f ) \n",hit_point.x,hit_point.y,hit_point.z );
 

  atomicAdd(&params.planeHitCounter[0], 1);
}



extern "C" __global__ void __closesthit__prog_sphere() {
  unsigned int tri_id = optixGetPrimitiveIndex();
  // We defined out geometry as a triangle geometry. In this case the
  // We add the t value of the intersection
 

  unsigned int ray_id_payload = optixGetPayload_0();
  float ray_tmax =  optixGetRayTmax();

  float3 ray_dir = optixGetWorldRayDirection();
  float3 ray_origin = optixGetWorldRayOrigin();
  float3 hit_point = ray_origin + ray_tmax * ray_dir;

  printf( " hit sphere at ( %f, %f, %f ) \n",hit_point.x,hit_point.y,hit_point.z );
  atomicAdd(&params.sphereHitCounter[0], 1);

}



// extern "C" __global__ void __miss__prog() {}
// extern "C" __global__ void __anyhit__prog() {}
