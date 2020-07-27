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
#pragma once
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <optix.h>
#include <sstream>
#include <string>
#include <vector>

void read_volume(std::vector<float3> &vertices, std::vector<uint3> &triangles,
                 const std::string &filename) {
  int n_vertices, n_tets;
  std::string word;
  std::ifstream file(filename);
  std::string line;
  std::getline(file, line);
  std::stringstream strst(line);
  strst >> word;
  strst >> n_vertices;
  std::cout << word << "   " << n_vertices << "  \n";
  for (int i = 0; i < (n_vertices / 3); i++) {
    std::getline(file, line);
    strst.str(line);
    float3 point;
    strst >> point.x;
    strst >> point.y;
    strst >> point.z;
    vertices.push_back(point);
    strst >> point.x;
    strst >> point.y;
    strst >> point.z;
    vertices.push_back(point);
    strst >> point.x;
    strst >> point.y;
    strst >> point.z;
    vertices.push_back(point);
  }
  std::getline(file, line);
  strst.str(line);
  strst >> word;
  strst >> n_tets;
  std::cout << word << "   " << n_tets << "  \n";
  for (int i = 0; i < n_tets; i++) {
    std::getline(file, line);
    int n, x, y, z, w;
    strst.str(line);
    strst >> n;
    strst >> x;
    strst >> y;
    strst >> z;
    strst >> w;
    triangles.push_back(make_uint3(x, y, z));
    triangles.push_back(make_uint3(x, y, w));
    triangles.push_back(make_uint3(y, z, w));
    triangles.push_back(make_uint3(x, z, w));
  }
}
