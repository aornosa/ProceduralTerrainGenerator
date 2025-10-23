#include "tmath.hpp"

// Math functions
double clamp(double value, double min, double max) {
	return std::max(min, std::min(max, value));
}
glm::vec2 randomGradient(int ix, int iy) {
	const unsigned w = 8 * sizeof(unsigned);
	const unsigned s = w / 2; // rotation width
	unsigned a = ix, b = iy;
	a *= 3284157443;

	b ^= a << s | a >> (w - s);
	b *= 1911520717;

	a ^= b << s | b >> (w - s);
	a *= 2048419325;

	float random = a * (PI / ~(~0u >> 1));
	return glm::vec2(cos(random), sin(random));
}
double dotGridGradient(int ix, int iy, double x, double y) {
	glm::vec2 gradient = randomGradient(ix, iy);

	// Clculate the distance vector
	double dx = x - (double)ix;
	double dy = y - (double)iy;

	// Compute and return the dot-product
	return (dx * gradient.x + dy * gradient.y);
}
double interpolate(double a0, double a1, double w) {
	// Use smoothstep interpolation
	return (a1 - a0) * (3.0 - w * 2.0) * w * w + a0;
}


glm::vec3 randomGradient3D(int ix, int iy, int iz) {
	// Use a hash to get consistent pseudo-random numbers
	int h = ix * 374761393 + iy * 668265263 + iz * 2147483647;
	h = (h ^ (h >> 13)) * 1274126177;
	h = h ^ (h >> 16);

	// Map hash to [-1,1] for each component
	float x = ((h & 0xFF) / 255.0f) * 2.0f - 1.0f;
	float y = (((h >> 8) & 0xFF) / 255.0f) * 2.0f - 1.0f;
	float z = (((h >> 16) & 0xFF) / 255.0f) * 2.0f - 1.0f;

	glm::vec3 g(x, y, z);
	return glm::normalize(g); // normalize to unit vector
}

// Dot product between gradient and offset vector
double dotGridGradient3D(int ix, int iy, int iz, double x, double y, double z) {
	// Gradient at grid corner
	glm::vec3 g = randomGradient3D(ix, iy, iz);

	// Vector from grid corner to point
	glm::vec3 d(float(x - ix), float(y - iy), float(z - iz));

	// Dot product
	return glm::dot(g, d);
}