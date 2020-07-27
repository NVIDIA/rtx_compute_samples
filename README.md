# Overview
The purpose of these code examples is to show how to use ray tracing for computations and discuss some application patterns that could benefit from ray tracing frameworks. These samples do not perform any graphical rendering. The individual samples are described below.

Prior knowledge of basic ray tracing terminology is required. However, in-depth knowledge and experience with ray tracing is not needed.
Please refer to the *Optix* ray tracing docs: https://raytracing-docs.nvidia.com/optix7/index.html

The following samples are based on the *Optix 7* API. Old *Optix 6* based samples are in legacy-optix-6 branch in this repository and will no longer be maintained.

## optixSaxpy ##
**CUDA/Optix buffer interop.**

This sample shows how to work with CUDA allocated memory buffers and Optix in order to compute a simple *saxpy* operation in the [ray generation program](https://raytracing-docs.nvidia.com/optix6/guide_6_5/index.html#programs#ray-generation-programs). There are no rays traced in the code and no geometry is created. This example is useful to understand the Optix API and code structure.

## optixProjection ##
**CAD geometry/cartesian mesh mapping.**

This sample shows a 2D projection of a 3D mesh onto 6 planes along the coordinate axes. To compute individual projections, the program shoots rays at a fixed sampling interval (dx,dy) perpendicular to the plane of projection. Projections are also used to find the distance to the 3D mesh object.

## optixMeshRefine ##
**Cartesian mesh refinement.**

This sample shows how to identify cells that need refinement in a 3D structured mesh. We start with a coarse 3D structured grid overlapping a triangle geometry. Rays are used to identify cells that contain geometry and are tagged for refinement. This process can be repeated for multi stage mesh refinement.

## optixUniqPaths ##
**Finding unique bounce paths of rays in a complex geometry.**

This sample shows how to determine the set of unique paths (scattering sequences) through a complex geometry. We send a number of rays into a geometry consisting of perfect reflectors read from a file (in case of the sample, a sphere and two planes). For each ray, the sequence of reflections is recorded. Once all the sequences have been determined, a set of unique paths is constructed. 


## optixRayScattering ##
**Scattering rays by reflection or refraction.**

This sample shows how to generate rays and bounce them across different surfaces. Rays are reflected between different surfaces and the rays keep bouncing until *max_bounce* is reached. Ray energy can be damped similar to signal propagation, rays can be refracted, and energy can be deposited at different hit points. This is a typical use case in wave propagation.

## optixVolumeSampling ##
**Volume sampling techniques.**

This sample reads a simple 3D volume. The purpose here is to collect values when a ray hits a geometry along the direction of the ray. Two approaches are shown in this example. In the first approach, a value is accumulated for every hit point the ray encounters, which can be specific to that triangle or geometry primitive. In the second approach, rays of different lengths are shot from the same origin and then do regular sampling of the 3D volume.

## optixParticleCollision ##
**Detecting Particle Collisions with Geometry**

This sample reads a simple obj file to create a geometry. It then creates a list of particles with positions inside or around the geomtery and random velocity values. Using particle positions as the center of a ray and direction along the velocity, rays are then shot with infinite length. If a ray hits some geometry, then the distance between the position of the particle and geometry is recorded.

# Building

Use CMake (>=3.5) for building. 

```
mkdir build && cd build
cmake ../ -DOPTIX_HOME=<path to optix 7>
make
```

# Runtime files

The executables require runtime files, which include pre-compiled PTX files and *.obj* mesh files. These are copied by the build system into each example's binary directory. For example:

```
cd optixRayScattering/
# Note the `optixRayScatteringPrograms.ptx` and `sphereAndPlane.obj` files. 
```


## Running an example:

```
cd optixRayScattering/
./rayscattering
```
