#include <windows.h>

#include <GL/glew.h>
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
const float terrainVertexSpacing = 1.f;		// Only applies to X and Z axis
const int seed = 12345;


// FUNCTION DECLARATIONS
GLFWwindow* initOpenGL();

// RENDERING
void UpdateCamera(GLuint shaderProgram, glm::vec3 cameraPos);
void CreateVBOVAO(GLuint& VBO, GLuint& VAO, const float* vertices, size_t vertexCount); // Add Vertex Array Object to new Vertex Buffer Object using data from vertices array

// CALLBACKS
void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);	// Callback for keyboard input
void MouseMoveCallback(GLFWwindow* window, double xpos, double ypos);					// Callback for mouse movement input
void MouseScrollCallback(GLFWwindow* window, double xpos, double ypos);					// Callback for mouse scroll wheel input
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);			// Callback for mouse button input

// TERRAIN GENERATION
double perlinNoise2D(double x, double y);
void generateTerrainMesh(std::vector<GLfloat> &vertices, std::vector<GLuint> &indices, double (*noise_fun)(double, double));	// Generate terrain mesh using Perlin noise

// MATH
double clamp(double value, double min, double max);


int main() {
	// Initialize OpenGL and create window
	GLFWwindow* window = initOpenGL();

	// Declare terrain data structures
	std::vector<GLfloat> terrainVertices;
	std::vector<GLuint> terrainIndices;

	GLuint terrainVBO, terrainVAO, terrainEBO; // Vertex Buffer Object, Vertex Array Object, Element Buffer Object

	// enerate terrain mesh
	generateTerrainMesh(terrainVertices, terrainIndices, perlinNoise2D);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glBindVertexArray(terrainVAO);


		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

double perlinNoise2D(double x, double y) {
	// Placeholder for Perlin noise function
	return (sin(x) + cos(y)) * 0.5;
}

void generateTerrainMesh(std::vector<GLfloat>& vertices, std::vector<GLuint> &indices, double (*noise_fun)(double, double)) {
	vertices.clear();
	indices.clear();

	vertices.reserve(terrainGridSize * terrainGridSize * 8);
	indices.reserve((terrainGridSize - 1) * (terrainGridSize - 1) * 6);

	// Precompute heights
	std::vector<double> heights((size_t)terrainGridSize*terrainGridSize);
	for (int z = 0; z < terrainGridSize; ++z) {
		for (int x = 0; x < terrainGridSize; ++x) {
			double worldX = (x - terrainGridSize / 2.0) * terrainVertexSpacing;
			double worldZ = (z - terrainGridSize / 2.0) * terrainVertexSpacing;
			double h = noise_fun(worldX, worldZ); // Scale later using amplitude and frequency as noise_fun(worldX*freq, worldZ*freq)*amplitude;
			heights[z * terrainGridSize + x] = h;
		}
	}

	// Build vertices with normals 
	for (int z = 0; z < terrainGridSize; ++z) {
		for (int x = 0; x < terrainGridSize; ++x) {
			double h = heights[z * terrainGridSize + x];

			double worldX = (x - terrainGridSize / 2.0) * terrainVertexSpacing;
			double worldZ = (z - terrainGridSize / 2.0) * terrainVertexSpacing;

			// Calculate normals using central differences
			double heightL = (x > 0) ? heights[z * terrainGridSize + (x - 1)] : h;
			double heightR = (x < terrainGridSize - 1) ? heights[z * terrainGridSize + (x + 1)] : h;
			double heightD = (z > 0) ? heights[(z - 1) * terrainGridSize + x] : h;
			double heightU = (z < terrainGridSize - 1) ? heights[(z + 1) * terrainGridSize + x] : h;

			glm::vec3 normal = glm::normalize(glm::vec3(heightL - heightR, 2.0f, heightD - heightU));

			// Tex coords for tiling texture (use later)
			float u = (float)x / (terrainGridSize - 1);
			float v = (float)z / (terrainGridSize - 1);

			// Push position data
			vertices.push_back((float)worldX);
			vertices.push_back((float)h);
			vertices.push_back((float)worldZ);

			// Push normal data
			vertices.push_back(normal.x);
			vertices.push_back(normal.y);
			vertices.push_back(normal.z);

			/*
			// Push tex coord data
			vertices.push_back(u);
			vertices.push_back(v);
			*/
		}
	}

	// Build indices for triangle strips 
	/*
		The triangles are built getting a square of 4 vertices from the grid, dividing it into 2 triangles.
		All triangles will have a shape like this:

		(z, x)--(z+1,x+1)
		   |  \      |
		   |   \ T2  |
		   |    \    |
		   | T1  \   |
		   |      \  |
		(z+1,x)--(z, x+1)
	*/
	for (int z = 0; z < terrainGridSize; ++z) {
		for (int x = 0; x < terrainGridSize; ++x) {
			GLuint topLeft = z * terrainGridSize + x;
			GLuint topRight = topLeft + 1;
			GLuint bottomLeft = (z + 1) * terrainGridSize + x;
			GLuint bottomRight = bottomLeft + 1;

			//T1
			indices.push_back(topLeft);
			indices.push_back(bottomLeft);
			indices.push_back(bottomRight);

			//T2
			indices.push_back(topLeft);
			indices.push_back(bottomRight);
			indices.push_back(topRight);
		}
	}
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