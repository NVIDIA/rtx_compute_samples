# optixParticleCollision #
 **Detecting Particle Collisions with Geometry**

 This sample reads a simple obj file and builds an acceleration structure for it. A list of particles with positions inside or around the geometry and random velocity values are initialized. Using particle positions as the origin of a ray and direction along the velocity, rays are then with infinite length. If a ray hits geometry, the distance between the position of the particle and geometry is recorded and the particle position is updated accordingly. In a particle simulation, the maximum trace distance could be adjusted to respect the time step and recursive tracing could be used to handle scattering events.

## Building

 Use CMake (>=3.17) for building (required for native CUDA support).

### Linux

```bash
 mkdir build && cd build
 cmake .. -DOPTIX_ROOT=<path/to/optix>
 cmake --build .
```

### Windows

```bash
 mkdir build && cd build
 cmake -DOPTIX_ROOT=<path/to/optix> ..
 cmake --build .
```

## Running the Sample

Be aware that the sample relies on the PTX files and the `cow.obj` to be available at specific paths relative to the current working directory. The PTX must be located in `./Release/ptx` and `cow.obj` must be in `../`.

### Linux 

```bash
cd build
Release/optixParticleCollision [<num_particles] [<output_file.csv>]
```

### Windows 

```bash
cd build
Release\optixParticleCollision.exe [<num_particles] [<output_file.csv>]
```
