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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <iomanip>
#include <iostream>
#include <iterator>

#include <maxBounce.h>
#include <thrustSortPaths.h>

struct Path {
  int path[MAX_BOUNCE];
};

void print_paths_uniq(thrust::device_vector<int> &,
                      thrust::device_vector<Path> &);

struct pathCmp
    : public thrust::binary_function<const Path &, const Path &, bool> {
  __host__ __device__ bool operator()(const Path &p1, const Path &p2) const {
    int i = 0;
    while (i <= p1.path[0]) {
      if (p1.path[i] < p2.path[i])
        return false;
      if (p1.path[i] > p2.path[i])
        return true;
      i++;
    }

    return false;
  }
};

// inequality functor used in inner product for counting the number of
// independent paths
struct notEqualToInt
    : public thrust::binary_function<const Path &, const Path &, int> {
  __host__ __device__ int operator()(const Path &p1, const Path &p2) const {
    int i = 0;
    while (i <= p1.path[0]) {
      if (p1.path[i] != p2.path[i])
        return 1;
      i++;
    }
    return 0;
  }
};

// equality functor used to in reduce by key
struct equalTo : public thrust::equal_to<const Path &> {
  __host__ __device__ bool operator()(const Path &p1, const Path &p2) const {
    int i = 0;
    while (i <= p1.path[0]) {
      if (p1.path[i] != p2.path[i])
        return false;
      i++;
    }
    return true;
  }
};

void sort_paths_device(unsigned int count, void *bouncesPtr) {

  thrust::device_ptr<Path> bouncesTptr =
      thrust::device_pointer_cast(bouncesPtr);

  thrust::device_vector<Path> paths(bouncesTptr, bouncesTptr + count);

  /*
  // in case we want to print out the device vector
  //
  thrust::host_vector<Path> t;
  t = paths;
  for(thrust::host_vector<Path>::iterator p = t.begin(); p<t.end(); p++){
         for(int i=0; i<MAX_BOUNCE; i++)
                 printf("%d ", (*p).path[i]);
         printf("\n");
  }
  */

  // sort the list of paths.
  thrust::sort(paths.begin(), paths.end(), pathCmp());

  // the inner product of the vector with itself shifted by one will count the
  // number different bins.
  int nBins =
      thrust::inner_product(paths.begin(), paths.end() - 1, paths.begin() + 1,
                            1, thrust::plus<int>(), notEqualToInt());

  printf("Unique paths : %d\n", nBins);

  // create storage for the unique paths
  thrust::device_vector<Path> pathsUniq(nBins);
  thrust::device_vector<int> nPathUniq(nBins);

  // Compress sorted list into bins of identical values
  thrust::reduce_by_key(paths.begin(), paths.end(),
                        thrust::constant_iterator<int>(1), pathsUniq.begin(),
                        nPathUniq.begin(), equalTo());

  print_paths_uniq(nPathUniq, pathsUniq);
}

typedef thrust::tuple<int &, Path &> PathTuple;

struct printPathTuple {
  __host__ __device__ void operator()(const PathTuple &p) const {
    printf("%d : ", p.get<0>());
    if (p.get<1>().path[0] == 0)
      printf(" misses ");
    for (int i = 1; i <= p.get<1>().path[0]; i++)
      printf(" %d", p.get<1>().path[i]);
    printf("\n");
  }
};

void print_paths_uniq(thrust::device_vector<int> &nPathUniq,
                      thrust::device_vector<Path> &pathsUniq) {

  int nBins = pathsUniq.size();

  thrust::host_vector<int> nPathUniqHost(nBins);
  thrust::host_vector<Path> pathsUniqHost(nBins);

  // transfer unique paths and magnitude to host
  thrust::copy(pathsUniq.begin(), pathsUniq.end(), pathsUniqHost.begin());
  thrust::copy(nPathUniq.begin(), nPathUniq.end(), nPathUniqHost.begin());

  // iterate through list of unique paths and print magnitude
  thrust::for_each(thrust::host,
                   thrust::make_zip_iterator(make_tuple(nPathUniqHost.begin(),
                                                        pathsUniqHost.begin())),
                   thrust::make_zip_iterator(
                       make_tuple(nPathUniqHost.end(), pathsUniqHost.end())),
                   printPathTuple());
}

