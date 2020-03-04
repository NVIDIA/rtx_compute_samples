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

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <random>
#include <vector>

using namespace optix;

void saxpy_optix(int N, float a, float *dx, float *dy) {
  optix::Context context = optix::Context::create();
  context->setEntryPointCount(1);
  context->setRayTypeCount(1);
  context->setRayGenerationProgram(
      0, context->createProgramFromPTXFile("kernels.ptx", "RayGeneration"));

  optix::Buffer xbuff =
      context->createBufferForCUDA(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, N);
  xbuff->setDevicePointer(0, (void *)dx);

  context["x"]->setBuffer(xbuff);

  optix::Buffer ybuff =
      context->createBufferForCUDA(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, N);
  ybuff->setDevicePointer(0, (void *)dy);

  context["y"]->setBuffer(ybuff);
  context["a"]->setFloat(a);

  context->launch(0, N);

  return;
}
