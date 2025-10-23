#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>

#define PI 3.14159265358979323846

// MATH
double clamp(double value, double min, double max);
/*
Produces the dot product of the distance and gradient vectors.
*/
double dotGridGradient(int ix, int iy, double x, double y);
/*
Interpolation function (smoothstep) optimized for perlin.
*/
double interpolate(double a0, double a1, double w);
/*
Linear interpolation between a and b with t in [0, 1].
a and b are the values to interpolate between,
t is the interpolation factor.
*/
double lerp(double a, double b, double t);
/*
Smoother interpolation between a and b with t in [0, 1].
a and b are the values to interpolate between,
t is the interpolation factor.
*/
double slerp(double a, double b, double t);

glm::vec2 randomGradient(int ix, int iy);