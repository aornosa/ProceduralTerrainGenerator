#include "noise.hpp"

// 2D
double voronoiNoise2D(double x, double y, int seed, int numPoints)
{
	// Grid cell coordinates
	int cx = (int)floor(x);
	int cy = (int)floor(y);

	// Limit number of points
	if (numPoints <= 0) numPoints = 1;

	// Search neighboring cells for closest point
	double minDist = DBL_MAX;		// Set to max

	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			// Neighbor cell coordinates
			int nx = cx + dx;
			int ny = cy + dy;

			// Generate pseudo-random points in cell based on seed
			std::mt19937 generator((nx * 73856093) ^ (ny * 19349663) ^ (seed * 83492791));
			std::uniform_real_distribution<double> distribution(0.0, 1.0);

			for (int i = 0; i < numPoints; ++i) {
				// Random point in cell
				double pointX = (double)nx + distribution(generator);
				double pointY = (double)ny + distribution(generator);

				// Compute distance to point
				double ddx = pointX - x;
				double ddy = pointY - y;
				double dist = sqrt(ddx * ddx + ddy * ddy);

				// Update minimum distance
				if (dist < minDist) {
					minDist = dist;
				}
			}
		}
	}

	return minDist;
}

double perlinNoise2D(double x, double y, int seed)
{
	// Grid cell coordinates
	int x0 = (int)floor(x);
	int y0 = (int)floor(y);
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	// Compute sampled interpolation weights
	double sx = x - (double)x0;
	double sy = y - (double)y0;

	// Compute dot products
	// Top corners
	double n0 = dotGridGradient(x0, y0, x, y);
	double n1 = dotGridGradient(x1, y0, x, y);
	double ix0 = interpolate(n0, n1, sx);		// Interpolate horizontally

	// Bottom corners
	n0 = dotGridGradient(x0, y1, x, y);
	n1 = dotGridGradient(x1, y1, x, y);
	double ix1 = interpolate(n0, n1, sx);		// Interpolate horizontally

	// Interpolate vertically
	return interpolate(ix0, ix1, sy);
}

double fractalNoise2D(double (*noiseFunc2D)(double, double, int), double x, double y, int octaves, int seed, float frequency, double persistence, double amplitude)
{
	double total = 0.0;
	double maxValue = 0.0; // Used for normalizing result to [0,1]
	for (int i = 0; i < octaves; ++i) {
		total += noiseFunc2D(x * frequency, y * frequency, seed) * amplitude;
		maxValue += amplitude;
		amplitude *= persistence;
		frequency *= 2.0;
	}
	return total / maxValue; // Normalize to [0,1]
}


// 3D
double voronoiNoise3D(double x, double y, double z, int seed, int numPoints)
{
	// Grid cell coordinates
	int cx = (int)floor(x);
	int cy = (int)floor(y);
	int cz = (int)floor(z);

	// Limit number of points
	if (numPoints <= 0) numPoints = 1;

	// Search neighboring cells for closest point
	double minDist = DBL_MAX;		// Set to max

	for (int dz = -1; dz <= 1; ++dz) {
		for (int dy = -1; dy <= 1; ++dy) {
			for (int dx = -1; dx <= 1; ++dx) {
				// Neighbor cell coordinates
				int nx = cx + dx;
				int ny = cy + dy;
				int nz = cz + dz;

				// Generate pseudo-random points in cell based on seed
				std::mt19937 generator((nx * 73856093) ^ (ny * 19349663) ^ (nz * 83492791) ^ (seed * 1234567));
				std::uniform_real_distribution<double> distribution(0.0, 1.0);

				for (int i = 0; i < numPoints; ++i) {
					// Random point in cell
					double pointX = (double)nx + distribution(generator);
					double pointY = (double)ny + distribution(generator);
					double pointZ = (double)nz + distribution(generator);

					// Compute distance to point
					double ddx = pointX - x;
					double ddy = pointY - y;
					double ddz = pointZ - z;
					double dist = sqrt(ddx * ddx + ddy * ddy + ddz * ddz);

					// Update minimum distance
					if (dist < minDist) {
						minDist = dist;
					}
				}
			}
		}
	}
	
	return minDist;
}

double voronoiNoise3D(double x, double y, double z, int seed) {
	return voronoiNoise3D(x, y, z, seed, DEFAULT_VORONOI_POINTS);
}

double perlinNoise3D(double x, double y, double z, int seed)
{
	// Grid cell coordinates
	int x0 = (int)floor(x);
	int x1 = x0 + 1;
	int y0 = (int)floor(y);
	int y1 = y0 + 1;
	int z0 = (int)floor(z);
	int z1 = z0 + 1;

	// Fractional part
	double sx = x - x0;
	double sy = y - y0;
	double sz = z - z0;

	// Dot products at cube corners
	double n000 = dotGridGradient3D(x0, y0, z0, x, y, z);
	double n100 = dotGridGradient3D(x1, y0, z0, x, y, z);
	double n010 = dotGridGradient3D(x0, y1, z0, x, y, z);
	double n110 = dotGridGradient3D(x1, y1, z0, x, y, z);
	double n001 = dotGridGradient3D(x0, y0, z1, x, y, z);
	double n101 = dotGridGradient3D(x1, y0, z1, x, y, z);
	double n011 = dotGridGradient3D(x0, y1, z1, x, y, z);
	double n111 = dotGridGradient3D(x1, y1, z1, x, y, z);

	// Interpolate along x
	double ix00 = interpolate(n000, n100, sx);
	double ix10 = interpolate(n010, n110, sx);
	double ix01 = interpolate(n001, n101, sx);
	double ix11 = interpolate(n011, n111, sx);

	// Interpolate along y
	double iy0 = interpolate(ix00, ix10, sy);
	double iy1 = interpolate(ix01, ix11, sy);

	// Interpolate along z
	return interpolate(iy0, iy1, sz);
}

double fractalNoise3D(double (*noiseFunc3D)(double, double, double, int), double x, double y, double z, int octaves, int seed, float frequency, double persistence, double amplitude)
{
	double total = 0.0;
	double maxValue = 0.0; // Used for normalizing result to [0,1]

	for (int i = 0; i < octaves; ++i) {
		total += noiseFunc3D(x * frequency, y * frequency, z * frequency, seed) * amplitude;
		maxValue += amplitude;
		amplitude *= persistence;
		frequency *= 2.0;
	}

	return total / maxValue; // Normalize to [0,1]
}