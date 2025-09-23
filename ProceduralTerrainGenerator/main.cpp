#include <windows.h>

#include <GL/GL.h>
#include <GL/GLU.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

// SHADER SOURCES

// RUNTIME VARIABLES
bool isPaused = false;

// SETTINGS
GLint screenWidth = 800;
GLint screenHeight = 600;

// CAMERA
glm::vec3 cameraPos = glm::vec3(0.0f, 5.0f, 20.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

// TERRAIN SETTINGS
const int terrainGridSize = 128;

// FUNCTION DECLARATIONS
GLFWwindow* initOpenGL();
double perlinNoise2D(double x, double y);


int main() {
	GLFWwindow* window = initOpenGL();
	while (!glfwWindowShouldClose(window)) {



		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

double perlinNoise2D(double x, double y) {
	// Placeholder for Perlin noise function
	return (sin(x) + cos(y)) * 0.5;
}

GLFWwindow* initOpenGL()
{
	if (!glfwInit()) {
		std::cerr << "Failed to start GLFW" << std::endl;
		return nullptr;
	}

	GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "Procedural Terrain Generator", NULL, NULL);
	
	return window;
}