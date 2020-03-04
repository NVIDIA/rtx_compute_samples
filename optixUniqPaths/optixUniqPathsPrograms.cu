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

struct Payload {
  int rayId;    // unique id of the ray
  float tPath;  // total lenth of the path with multiple bounces
  int *bounces; // array with ids of the scatterrs along the path
};

// context variables declared in loadMesh
rtDeclareVariable(rtObject, topObject, , );
rtDeclareVariable(float3, delta, , );
rtDeclareVariable(float3, minCorner, , );
rtDeclareVariable(float3, maxCorner, , );
rtDeclareVariable(int, MAX_BOUNCE, , );

// semantic variables used durig tracing
rtDeclareVariable(uint3, launchIdx, rtLaunchIndex, );
rtDeclareVariable(uint3, launchDim, rtLaunchDim, );
rtDeclareVariable(Payload, payload, rtPayload, );
rtDeclareVariable(Ray, currRay, rtCurrentRay, );

// buffer objects
rtBuffer<float, 1> tPath;
rtBuffer<int, 1> bounces;
rtBuffer<float3, 1> vertexBuf;
rtBuffer<uint3, 1> triangleBuf;

RT_PROGRAM void rayGen() {

  float xo = minCorner.x + delta.x * launchIdx.x;
  float yo = minCorner.y + delta.y * launchIdx.y;
  float zo = minCorner.z;

  // setting the per ray data (payload)
  Payload pld;
  pld.tPath = 0.0f;
  pld.rayId = launchIdx.x + launchDim.x * launchIdx.y;
  pld.bounces = &bounces[pld.rayId * MAX_BOUNCE];
  pld.bounces[0] = 0;

  // create a ray
  float3 rayOrigin = make_float3(xo, yo, zo);
  float3 rayDir = normalize(make_float3(0.0, 0.0, 1.0));
  Ray ray = make_Ray(rayOrigin, rayDir, 0, 0, RT_DEFAULT_MAX);

  // launch the ray
  rtTrace(topObject, ray, pld, RT_VISIBILITY_ALL, RT_RAY_FLAG_NONE);

  // store the result and  number of bounces back to the global buffers
  tPath[launchIdx.x + launchDim.x * launchIdx.y] = pld.tPath;
}

RT_PROGRAM void exception() { rtPrintExceptionDetails(); }

// TODO: to improve performance, pre-compute and pack the normals.
// but here we compute them while tracing
__device__ __forceinline__ float3 outwardNormal(const unsigned int triId,
                                                const float3 rayDir) {
  uint3 vids = triangleBuf[triId];
  float3 v1 = vertexBuf[vids.x];
  float3 v2 = vertexBuf[vids.y];
  float3 v3 = vertexBuf[vids.z];
  float3 normal = cross((v2 - v1), (v3 - v1));

  // check if the normal is facing in opposite direction of the ray dir, if yes
  // flip it in Z dir
  if (normal.z * rayDir.z < 0) {
    normal.z *= -1.0f;
  }
  return normal;
}

RT_PROGRAM void closest_hit() {
  unsigned int triId = rtGetPrimitiveIndex();
  // We defined out geometry as a triangle geometry. In this case the
  // currRay.tmax contains the t value of the intersection
  payload.tPath += currRay.tmax;
  int nBounce = ++payload.bounces[0];
  payload.bounces[nBounce] = triId;

  // report individual bounces
  // printf("Ray = %d, bounce = %d, id =%d \n", payload.rayId, triId,
  // payload.bounces[0]);

  if (nBounce < MAX_BOUNCE - 1) {
    float3 outNormal = outwardNormal(triId, currRay.direction);
    float3 reflectDir = reflect(currRay.direction, outNormal);
    float3 hitPoint = currRay.origin + currRay.tmax * currRay.direction;

    // Minimal distance the ray has to travel to report next hit
    float tmin = 1e-5;

    Ray reflectRay(hitPoint, reflectDir, 0, tmin, RT_DEFAULT_MAX);
    // contine tracing the ray in reflected direction
    rtTrace(topObject, reflectRay, payload, RT_VISIBILITY_ALL,
            RT_RAY_FLAG_NONE);
  }
}

RT_PROGRAM void any_hit() {
  // if (payload.nBounce >= MAX_BOUNCE) {
  if (payload.bounces[0] >= MAX_BOUNCE - 1) {
    rtTerminateRay();
  }
}

RT_PROGRAM void miss() {}
