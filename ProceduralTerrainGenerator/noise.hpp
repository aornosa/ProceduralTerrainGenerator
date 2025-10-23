#include <vector>
#include <random>

#include "tmath.hpp"

// 2D

/// <summary>
/// Worley implementation of 2D Voronoi noise
/// </summary>
/// <param name="x">X component of the noise to sample</param>
/// <param name="y">Y component of the noise to sample</param>
/// <param name="seed">Initial seed for randomness</param>
/// <param name="numPoints">Number of points of resolution</param>
/// <returns>Z equivalent component sampled at (x,y)</returns>
double voronoiNoise2D(double x, double y, int seed, int numPoints);

/// <summary>
/// Perlin implementation of 2D noise
/// </summary>
/// <param name="x">X component of the noise to sample</param>
/// <param name="y">Y component of the noise to sample</param>
/// <param name="seed">Initial seed for randomness</param>
/// <returns>Z component sampled at (x,y)</returns>
double perlinNoise2D(double x, double y, int seed);

/// <summary>
/// Fractal implementation of 2D noise
/// </summary>
/// <param name="noiseFunc2D">Pointer to the 2D noise function to use (x, y, seed)</param>
/// <param name="x">X component of the noise to sample</param>
/// <param name="y">Y component of the noise to sample</param>
/// <param name="frequency">Frequency of the noise</param>
/// <param name="octaves">Number of octaves to combine</param>
/// <param name="persistence">Persistence value for amplitude scaling</param>
/// <param name="seed">Initial seed for randomness</param>
/// <returns>Z component sampled at (x,y) of the stacked noise function</returns>
double fractalNoise2D(double (*noiseFunc2D)(double, double, int), double x, double y, int octaves, int seed, float frequency = 1.0, double persistence = 0.5, double amplitude = 1.0);

// 3D

/// <summary>
/// Voronoi implementation of 3D noise
/// </summary>
/// <param name="x">X component of the noise to sample</param>
/// <param name="y">Y component of the noise to sample</param>
/// <param name="z">Z component of the noise to sample</param>
/// <param name="seed">Initial seed for randomness</param>
/// <param name="numPoints">Number of points of resolution</param>
/// <returns>W component sampled at (x,y,z)</returns>
double voronoiNoise3D(double x, double y, double z, int seed, int numPoints);

/// <summary>
/// Perlin implementation of 3D noise
/// </summary>
/// <param name="x">X component of the noise to sample</param>
/// <param name="y">Y component of the noise to sample</param>
/// <param name="z">Z component of the noise to sample</param>
/// <param name="seed">Initial seed for randomness</param>
/// <returns>W component sampled at (x,y,z)</returns>
double perlinNoise3D(double x, double y, double z, int seed);

/// <summary>
/// Fractal implementation of 3D noise
/// </summary>
/// <param name="noiseFunc3D">Pointer to the 3D noise function to use (x, y, z, seed)</param>
/// <param name="x">X component of the noise to sample</param>
/// <param name="y">Y component of the noise to sample</param>
///	<param name="z">Z component of the noise to sample</param>
/// <param name="frequency">Frequency of the noise</param>
/// <param name="octaves">Number of octaves to combine</param>
/// <param name="persistence">Persistence value for amplitude scaling</param>
/// <param name="seed">Initial seed for randomness</param>
/// <returns>W component sampled at (x,y,z) of the stacked noise function</returns>
double fractalNoise3D(double (*noiseFunc3D)(double, double, double, int), double x, double y, double z, int octaves, int seed, float frequency = 1.0, double persistence = 0.5, double amplitude = 1.0);