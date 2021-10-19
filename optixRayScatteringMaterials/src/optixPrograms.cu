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
   // we need to set the stride such that buildInput[1] uses the next SBT entry
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
 
 // TODO: to improve performance, pre-compute and pack the normals.
 // but here we compute them while tracing
 __device__ __forceinline__ float3 outwardNormal(const unsigned int triId,
                                                 const float3 rayDir) {
 
   float3 vertex[3];
   OptixTraversableHandle gas_handle = optixGetGASTraversableHandle();
   optixGetTriangleVertexData(gas_handle, triId, 0, 0, vertex);
 
   float3 normal = cross((vertex[1] - vertex[0]), (vertex[2] - vertex[0]));
 
   // check if the normal is facing in opposite direction of the ray dir, if yes
   // flip it in Z dir
   if (normal.z * rayDir.z < 0) {
     normal.z *= -1.0f;
   }
   return normal;
 }
 
 extern "C" __global__ void __closesthit__reflector_refractor_prog() {
   unsigned int tri_id = optixGetPrimitiveIndex();
   // We defined out geometry as a triangle geometry. In this case the
   // We add the t value of the intersection
   float ray_tmax = optixGetRayTmax();
 
   unsigned int ray_id_payload = optixGetPayload_0();
   unsigned int tpath_payload = optixGetPayload_1();
 
   float total_path_length = ray_tmax + __uint_as_float(tpath_payload);
   optixSetPayload_1(__float_as_uint(total_path_length));
 
   // report individual bounces
   // printf("Ray = %d, pathlen = %f\n", payload.rayId, payload.tPath);
 
   float3 ray_dir = optixGetWorldRayDirection();
   float3 ray_ori = optixGetWorldRayOrigin();
 
   float3 out_normal = outwardNormal(tri_id, ray_dir);
   float3 reflect_dir = reflect(ray_dir, out_normal);
   float3 hit_point = ray_ori + ray_tmax * ray_dir;
 
   // Minimal distance the ray has to travel to report next hit
   float tmin = 1e-5;
   float tmax = params.delta.z + 100.0;
   float ray_time = 0.0f;
   OptixVisibilityMask visibilityMask = 255;
   unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
   unsigned int SBToffset = 0;
   unsigned int SBTstride = 0;
   unsigned int missSBTIndex = 0;
 
   optixTrace(params.handle, hit_point, reflect_dir, tmin, tmax, ray_time,
              visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
              ray_id_payload, tpath_payload);
 }

 extern "C" __global__ void __closesthit__absorber_prog() {
  unsigned int tri_id = optixGetPrimitiveIndex();
  // We defined out geometry as a triangle geometry. In this case the
  // We add the t value of the intersection
  float ray_tmax = optixGetRayTmax();

  unsigned int ray_id_payload = optixGetPayload_0();
  unsigned int tpath_payload = optixGetPayload_1();

  float total_path_length = ray_tmax + __uint_as_float(tpath_payload);
  optixSetPayload_1(__float_as_uint(total_path_length));

  // report individual absorbs
 //printf("Ray = %d, pathlen = %f has been absorbed  \n",ray_id_payload, tpath_payload);
 
}

 
 // extern "C" __global__ void __miss__prog() {}
 // extern "C" __global__ void __anyhit__prog() {}
 