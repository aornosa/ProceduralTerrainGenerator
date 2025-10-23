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
