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
#include <unordered_map>

// SHADER SOURCES
// Basic vertex shader
const char* vertexShaderSource = R"glsl(
	#version 330 core
	layout(location = 0) in vec3 aPos;		// Vertex position
	layout(location = 1) in vec3 aNormal;	// Vertex normal
	layout(location = 2) in vec2 aTexCoord; // Vertex texture coordinate
	
	out vec3 FragPos;
	out vec3 Normal;
	out vec2 TexCoord;
	
	uniform mat4 model;
	uniform mat4 view;
	uniform mat4 projection;
	
	void main() {
		FragPos = vec3(model * vec4(aPos, 1.0));
		Normal = mat3(transpose(inverse(model))) * aNormal;
		TexCoord = aTexCoord;
		gl_Position = projection * view * vec4(FragPos, 1.0);
})glsl";
//Basic fragment shader
const char* fragmentShaderSource = R"glsl(
	#version 330 core
	out vec4 FragColor;

	in vec3 FragPos;
	in vec3 Normal;
	in vec2 TexCoord;

	uniform vec3 lightPos;
	uniform vec3 viewPos;
	uniform sampler2D texture1;

	// Camera and fog uniforms
	uniform vec3 cameraPos;
	uniform vec3 fogColor = vec3(0.0, 0.0, 0.0);
	uniform float fogDensity = 0.0;
	uniform float maxFogDistance;
	uniform float minFogDistance;

	void main() {
		// Ambient Lighting
		float ambientStrength = 0.3;
		vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);

		// Diffuse Lighting
		vec3 norm = normalize(Normal);
		vec3 lightDir = normalize(lightPos - FragPos);
		float diff = max(dot(norm, lightDir), 0.0);
		vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

		// Specular Lighting
		float specularStrength = 0.5;
		vec3 viewDir = normalize(viewPos - FragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
		vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);

		vec3 result = (ambient + diffuse + specular) * texture(texture1, TexCoord).rgb;
		
		// Add fog
		if(fogDensity > 0.0) {
			float distance = length(FragPos - cameraPos);
			float fogFactor = 1.0 - exp(-distance * fogDensity);
			
			result = mix(result, fogColor, fogFactor); 
		}
		FragColor = vec4(result, 1.0);
})glsl";

// MATH CONSTANTS
const double PI = 3.14159265358979323846;

// RUNTIME VARIABLES
bool isPaused = false;

// SETTINGS
GLint screenWidth = 800;
GLint screenHeight = 600;

// CAMERA VARIABLES
glm::vec3 cameraPos = glm::vec3(0.0f, 5.0f, 20.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float yaw = -90.0f;	
float pitch = 0.0f;
float lastX = screenWidth / 2.0f;
float lastY = screenHeight / 2.0f;

// CAMERA SETTINGS
float cameraSpeed = 0.5f;		// Camera movement speed
float sensitivity = 0.1f;		// Mouse sensitivity


// TERRAIN SETTINGS
const int terrainGridSize = 64;
const float terrainVertexSpacing = 1.f;		// Only applies to X and Z axis
const float terrainHeightScale = 15.f;		// Scale height (Y axis)
const float terrainFrequency = 0.0005f;		// Frequency of the noise function
const int noiseOctaveN = 6;					// Number of noise layers
const int seed = 12345;						// IMPLEMENT RANDOM SEEDING

// SKY SETTINGS
const glm::vec4 skyColor = { 0.4f, 0.65f, 1.0f, 1.0f };

const float fogDensity = 0.0005;
//const float fogMinDistance; // Unused
//const float fogMaxDistance; // Unused
const glm::vec3 fogColor = { 0.4, 0.65, 1.0 };

// RENDER SETTINGS
const int renderDistance = 10;			// Render distance in chunks

// OBJECT DECLARATIONS
class Chunk;

// FUNCTION DECLARATIONS
GLFWwindow* initOpenGL();

// RENDERING
void UpdateCamera(GLuint shaderProgram, glm::vec3 cameraPos);
void CreateBufferArrayObjects(GLuint& VBO, GLuint& VAO, GLuint& EBO, const float* vertices, size_t vertexCount, const GLuint* indices, size_t indexCount); // Add Vertex Array Object to new Vertex Buffer Object using data from vertices array
void UpdateBufferArrayObjects(GLuint& VBO, GLuint& EBO, const float* vertices, size_t vertexCount, const GLuint* indices, size_t indexCount); // Update existing Vertex Buffer Object with new data from vertices array (Add new chunks)

// SHADING
GLuint CompileShaderProgram(const char* vertexSource, const char* fragmentSource); // Compile and link vertex and fragment shaders into a shader program

// CALLBACKS
void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);	// Callback for keyboard input
void MouseMoveCallback(GLFWwindow* window, double xpos, double ypos);					// Callback for mouse movement input
void MouseScrollCallback(GLFWwindow* window, double xpos, double ypos);					// Callback for mouse scroll wheel input
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);			// Callback for mouse button input

// TERRAIN GENERATION
/*
Function to sample 2D Perlin noise at coordinates (x, y).
Generates smooth, continuous noise values that can be used for terrain height mapping.
- Based Zipped's implementation in C++ -
*/
double perlinNoise2D(double x, double y);
/*
Layered Perlin noise function to create fractal noise.
Combines multiple octaves of Perlin noise to produce more complex and natural-looking terrain features.
*/
double fractalNoise2D(double x, double y, int octaves, double persistence); // Generate fractal noise by combining multiple octaves of Perlin noise
// TODO: REFACTOR NOISE FUNCTION INPUT
void generateTerrainMesh(std::vector<GLfloat> &vertices, std::vector<GLuint> &indices, double (*noise_fun)(double, double), glm::vec2 chunkPos);	// Generate terrain mesh using Perlin noise


// TERRAIN RENDERING
void updateVisibleChunks(std::unordered_map<long long, Chunk>& chunkMap, std::vector</*Chunk**/long long>& visibleChunks, glm::vec3 cameraPos, int renderDistance);


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



// TESTING
void genCube(std::vector<GLfloat>& vertices, std::vector<GLuint>& indices); // Generate a cube mesh for testing purposes



// OBJECT CLASSES
class Chunk {
public:
	// Properties
	glm::vec2 position;					// Position of the chunk in world space (X, Z)
	// CPU side
	std::vector<GLfloat> vertices;		// Vertex data (positions, normals, texture coordinates)
	std::vector<GLuint> indices;		// Index data for element drawing
	// GPU side
	GLuint VBO = 0, VAO = 0, EBO = 0;	// OpenGL buffer and array objects

	bool isInitialized;					// Flag to check if the chunk has been initialized
	bool isActive = false;				// Flag to check if the chunk is active (within render distance)
	Chunk() : position(glm::vec2(0.0f)), isInitialized(false) {}
	Chunk(glm::vec2 pos) : position(pos), isInitialized(false) {}

	~Chunk() {/*
		if (isInitialized) {
			glDeleteVertexArrays(1, &VAO);
			glDeleteBuffers(1, &VBO);
			glDeleteBuffers(1, &EBO);
		}*/
	}

	void initialize() {
		generateTerrainMesh(vertices, indices, perlinNoise2D, position);
		CreateBufferArrayObjects(VBO, VAO, EBO, vertices.data(), vertices.size(), indices.data(), indices.size());
		isInitialized = true;
		isActive = true;
	}

	void load() {
		if (isActive) return; // Already active
		if (!isInitialized) {
			initialize();
		}
		isActive = true;
	}

	void unload() {
		if (!isActive) return; // Already inactive
		isActive = false;
	}

	static long long hashCoords(int x, int y) {
		return (static_cast<long long>(static_cast<uint32_t>(x)) << 32) |
			static_cast<uint32_t>(y);
	}

	long long hash() const {
		return hashCoords(static_cast<int>(position.x), static_cast<int>(position.y));
	}
	
	int camToChunkCoord (int coord) const {
		return static_cast<int>(floor((coord) / (terrainGridSize * terrainVertexSpacing)));
	}
};


int main() {
	// Initialize OpenGL and create window
	GLFWwindow* window = initOpenGL();
	GLuint shaderProgram = CompileShaderProgram(vertexShaderSource, fragmentShaderSource);

	std::unordered_map<long long, Chunk> chunkMap; // Map to store chunks by their position key
	std::vector<long long> visibleChunks; // List of currently loaded chunk keys

	// Cube for testing
	std::vector<GLfloat> cubeVertices;
	std::vector<GLuint> cubeIndices;
	genCube(cubeVertices, cubeIndices);

	GLuint cubeVBO = 0, cubeVAO = 0, cubeEBO = 0;
	CreateBufferArrayObjects(cubeVBO, cubeVAO, cubeEBO, cubeVertices.data(), cubeVertices.size(), cubeIndices.data(), cubeIndices.size());


	// Create a simple 1x1 white texture so shader sampling is valid (CHANGE FOR TEXTURE GENERATION/SAMPLING)
	GLuint whiteTexture = 0;
	glGenTextures(1, &whiteTexture);
	glBindTexture(GL_TEXTURE_2D, whiteTexture);
	unsigned char whitePixel[3] = { 80, 255, 100 };
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, whitePixel);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	
	// Link shader program
	glUseProgram(shaderProgram);

	// Set fog values
	GLint fogDensityLoc = glGetUniformLocation(shaderProgram, "fogDensity");
	if (fogDensityLoc >= 0) glUniform1f(fogDensityLoc, fogDensity);

	GLint fogColorLoc = glGetUniformLocation(shaderProgram, "fogColor");
	if (fogColorLoc >= 0) glUniform3f(fogColorLoc, fogColor.r, fogColor.g, fogColor.b);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		UpdateCamera(shaderProgram, cameraPos);
		updateVisibleChunks(chunkMap, visibleChunks, cameraPos, renderDistance); // Update visible chunks based on camera position and render distance

		// TESTING RENDER CUBE
		glBindTexture(GL_TEXTURE_2D, whiteTexture);
		GLint texLoc = glGetUniformLocation(shaderProgram, "texture1");
		if (texLoc >= 0) glUniform1i(texLoc, 0);

		// Set light and view position uniforms
		GLint lightLoc = glGetUniformLocation(shaderProgram, "lightPos");
		if (lightLoc >= 0) glUniform3f(lightLoc, 10.0f, 10.0f, 10.0f);
		GLint viewLocP = glGetUniformLocation(shaderProgram, "viewPos");
		if (viewLocP >= 0) glUniform3f(viewLocP, cameraPos.x, cameraPos.y, cameraPos.z);

		// Model matrix for the cube
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, glm::vec3(0.0f, 1.0f, 0.0f)); // lift slightly so it's visible above y=0
		model = glm::scale(model, glm::vec3(2.0f)); // make it a bit bigger
		GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
		if (modelLoc >= 0) glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

		glBindVertexArray(cubeVAO);
		glDrawElements(GL_TRIANGLES, (GLsizei)cubeIndices.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		// Render visible chunks
		//std::cout << "Visible Chunks: " << visibleChunks.size() << "\n";
		for (long long keys : visibleChunks) {
			Chunk& chunk = chunkMap.at(keys);
			//std::cout << "Rendering Chunk at: (" << chunk.position.x << ", " << chunk.position.y << ")\n";
			if (!chunk.isInitialized) continue; // Safety check (skip if chunk not loaded yet)

			glm::mat4 model = glm::mat4(1.0f);
			model = glm::translate(model, glm::vec3(
				chunk.position.x * (terrainGridSize-1) * terrainVertexSpacing,
				0.0f,			  
				chunk.position.y * (terrainGridSize-1) * terrainVertexSpacing
			));
			GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
			if (modelLoc >= 0)
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

			glBindVertexArray(chunk.VAO);
			glDrawElements(GL_TRIANGLES, (GLsizei)chunk.indices.size(), GL_UNSIGNED_INT, 0);

			glBindVertexArray(0);
		}

		// Swap buffers and poll events (Frame main loop)
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

// RENDERING
void UpdateCamera(GLuint shaderProgram, glm::vec3 cameraPos) {
	glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)screenWidth / (float)screenHeight, 0.1f, 3000.0f);
	glUseProgram(shaderProgram);
	GLint cameraPosLoc = glGetUniformLocation(shaderProgram, "cameraPos");
	if (cameraPosLoc >= 0) glUniform3f(cameraPosLoc, cameraPos.x, cameraPos.y, cameraPos.z);

	GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
	GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
}
void CreateBufferArrayObjects(GLuint &VBO, GLuint &VAO, GLuint &EBO, const float *vertices, size_t vertexCount, const GLuint *indices, size_t indexCount) {
	glGenVertexArrays(1, &VAO); // Generate Vertex Array Object 
	glGenBuffers(1, &VBO);		// Generate Vertex Buffer Object
	glGenBuffers(1, &EBO);		// Generate Element Buffer Object

	glBindVertexArray(VAO);						// Bind Vertex Array Object
	glBindBuffer(GL_ARRAY_BUFFER, VBO);			// Bind Vertex Buffer Object
	glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(float), vertices, GL_STATIC_DRAW);		// Insert vertex data into Vertex Buffer Object

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO); // Bind Element Buffer Object
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(GLuint), indices, GL_STATIC_DRAW); // Insert index data into Element Buffer Object

	// Vertex attributes (n of floats reserved): pos(3), normal(3), texCoord(2). Total = 8 floats per vertex
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);					// Position attribute
	glEnableVertexAttribArray(0);																	// Enable position attribute

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))); // Normal attribute
	glEnableVertexAttribArray(1);																	// Enable normal attribute

	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float))); // TexCoord attribute (Textures)
	glEnableVertexAttribArray(2);																	// Enable texCoord attribute

	glBindVertexArray(0); // Unbind VAO
	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // Unbind EBO
}
void UpdateBufferArrayObjects(GLuint &VBO, GLuint &EBO, const float *vertices, size_t vertexCount, const GLuint *indices, size_t indexCount) {
	glBindBuffer(GL_ARRAY_BUFFER, VBO);														// Bind Vertex Buffer Object
	glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(float), vertices, GL_STATIC_DRAW);	// Update vertex data in Vertex Buffer Object

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);												// Bind Element Buffer Object
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(GLuint), indices, GL_STATIC_DRAW); // Update index data in Element Buffer Object
}

// SHADING
GLuint CompileShaderProgram(const char *vertexSource, const char *fragmentSource) {
	GLint success;												// Check for compilation errors

	// Vertex Shader
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);		// Create vertex shader
	glShaderSource(vertexShader, 1, &vertexSource, NULL);		// Attach vertex shader source code
	glCompileShader(vertexShader);								// Compile vertex shader
	
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);	// Check for compilation errors
	
	if (!success) {
		GLchar infoLog[512];
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// Fragment Shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);	// Create fragment shader
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);	// Attach fragment shader source code
	glCompileShader(fragmentShader);							// Compile fragment shader

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);	// Check for compilation errors

	if (!success) {
		GLchar infoLog[512];
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}


	// Shader Program
	GLuint shaderProgram = glCreateProgram();					// Create shader program
	glAttachShader(shaderProgram, vertexShader);				// Attach vertex shader to shader program
	glAttachShader(shaderProgram, fragmentShader);				// Attach fragment shader to shader program
	glLinkProgram(shaderProgram);								// Link shader program

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);	// Check for linking errors

	if (!success) {
		GLchar infoLog[512];
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}

	// Clean up (Shaders already linked in program)
	glDeleteShader(vertexShader);	// Delete vertex shader
	glDeleteShader(fragmentShader);	// Delete fragment shader

	return shaderProgram;			// Return shader program ID
}

// DRAWING
void DrawTerrain(GLuint shaderProgram, GLuint VAO, size_t indexCount) {
}

// CALLBACKS
void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods) { // TODO: Change input to continuous event instead of discrete key press
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true); // Close window on ESC key press
	if (key == GLFW_KEY_P && action == GLFW_PRESS)
		isPaused = !isPaused; // Toggle pause on P key press

	if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT) && !isPaused)
		cameraPos += cameraSpeed * cameraFront;
	if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT) && !isPaused)
		cameraPos -= cameraSpeed * cameraFront;
	if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT) && !isPaused)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT) && !isPaused)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (key == GLFW_KEY_SPACE && (action == GLFW_PRESS || action == GLFW_REPEAT) && !isPaused)
		cameraPos += cameraSpeed * cameraUp;
	if (key == GLFW_KEY_LEFT_SHIFT && (action == GLFW_PRESS || action == GLFW_REPEAT) && !isPaused)
		cameraPos -= cameraSpeed * cameraUp;
} 
void MouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	float xoffset = xpos - lastX;	// Calculate offset since last frame
	float yoffset = lastY - ypos;	// Reversed since y-coordinates go from bottom to top

	lastX = (float)xpos;			// Update lastX and lastY
	lastY = (float)ypos;

	xoffset *= sensitivity;			// Apply sensitivity
	yoffset *= sensitivity;

	yaw += xoffset;					// Update yaw and pitch
	pitch += yoffset;

	pitch = clamp(pitch, -89.0f, 89.0f);	// Constrain pitch

	glm::vec3 front;				// Calculate new front vector
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = glm::normalize(front);
}
void MouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	// Placeholder for mouse scroll handling (e.g., zooming)
}
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	// Placeholder for mouse button handling (e.g., selecting terrain points)
}

// MATH
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

// TERRAIN GENERATION
double perlinNoise2D(double x, double y) {
	// Placeholder for Perlin noise function

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
double fractalNoise2D(double x, double y, int octaves, double persistence) {
	double total = 0.0;
	double frequency = 1.0;
	double amplitude = 1.0;
	double maxValue = 0.0; // Used for normalizing result to [0,1]
	for (int i = 0; i < octaves; ++i) {
		total += perlinNoise2D(x * frequency, y * frequency) * amplitude;
		maxValue += amplitude;
		amplitude *= persistence;
		frequency *= 2.0;
	}
	return total / maxValue; // Normalize to [0,1]
}
void generateTerrainMesh(std::vector<GLfloat>& vertices, std::vector<GLuint> &indices, double (*noise_fun)(double, double), glm::vec2 chunkPos) {
	vertices.clear();
	indices.clear();

	int meshSize = terrainGridSize + 1;

	vertices.reserve(terrainGridSize * terrainGridSize * 8);
	indices.reserve((terrainGridSize - 1) * (terrainGridSize - 1) * 6);

	// Compute world offset of this chunk
	double chunkWorldX = chunkPos.x * (terrainGridSize-1) * terrainVertexSpacing;
	double chunkWorldZ = chunkPos.y * (terrainGridSize-1) * terrainVertexSpacing;

	// Precompute heights
	std::vector<double> heights((size_t)terrainGridSize*terrainGridSize);
	for (int z = 0; z < terrainGridSize; ++z) {
		for (int x = 0; x < terrainGridSize; ++x) {
			double worldX = chunkWorldX + x * terrainVertexSpacing;
			double worldZ = chunkWorldZ + z * terrainVertexSpacing;
			// TODO: USE SEVERAL OCTAVES OF NOISE TO GET MORE REALISTIC TERRAIN
			double h = pow(fractalNoise2D(worldX * terrainFrequency, worldZ * terrainFrequency, noiseOctaveN, 0.5) * terrainHeightScale,4); // Scale using amplitude and frequency as noise_fun(worldX*freq, worldZ*freq)*amplitude;
			heights[z * terrainGridSize + x] = h;
		}
	}

	// Build vertices with normals 
	for (int z = 0; z < terrainGridSize; ++z) {
		for (int x = 0; x < terrainGridSize; ++x) {
			double h = heights[z * terrainGridSize + x];

			double worldX = x * terrainVertexSpacing;
			double worldZ = z * terrainVertexSpacing;

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

			// Push tex coord data
			vertices.push_back(u);
			vertices.push_back(v);
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
	for (int z = 0; z < terrainGridSize-1; ++z) {
		for (int x = 0; x < terrainGridSize-1; ++x) {
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

// TERRAIN RENDERING
void updateVisibleChunks(std::unordered_map<long long, Chunk>& chunkMap, std::vector<long long/*Chunk* */>& visibleChunks, glm::vec3 cameraPos, int renderDistance) {
	// Determine current chunk coordinates based on camera position
	int currentChunkX = (int)floor(cameraPos.x / (terrainGridSize * terrainVertexSpacing));
	int currentChunkZ = (int)floor(cameraPos.z / (terrainGridSize * terrainVertexSpacing));

	// Remove chunks that are no longer within render distance (O(n) due to vector structure)
	visibleChunks.erase(
		std::remove_if(visibleChunks.begin(), visibleChunks.end(),
			[&](long long key) {		// Lambda to check if chunk is outside render distance (captured chunk coords)
				// Get chunk reference from the map
				auto it = chunkMap.find(key);
				if (it == chunkMap.end()) return true; // Chunk missing, remove from visibleChunks

				Chunk& chunk = it->second;

				int dx = chunk.position.x - currentChunkX;
				int dz = chunk.position.y - currentChunkZ;

				// Calculate if chunk is outside render distance
				if (dx*dx + dz*dz > renderDistance*renderDistance) {
					chunk.isActive = false;	// Mark chunk as inactive
					return true;				// Remove from visible chunks
				}
				return false;					// Keep in visible chunks
			}
		),
		visibleChunks.end()
	);

	// Load new chunks within render distance
	for (int dz = -renderDistance; dz <= renderDistance; ++dz) {
		for (int dx = -renderDistance; dx <= renderDistance; ++dx) {
			// Skip chunks outside circular render distance
			if (dx * dx + dz * dz > renderDistance * renderDistance) continue; 

			// Calculate current iteration chunk coordinates
			int chunkX = currentChunkX + dx;
			int chunkZ = currentChunkZ + dz;

			// Generate unique key for chunk coordinates
			long long chunkKey = Chunk::hashCoords(chunkX, chunkZ);

			auto it = chunkMap.find(chunkKey);
			if (it != chunkMap.end()) {
				if (it->second.isActive) continue;
				else {
					it->second.load();
					visibleChunks.push_back(chunkKey);
				}
			}
			else { // Create new chunk if it doesn't exist
				Chunk newChunk = Chunk(glm::vec2(chunkX, chunkZ));
				newChunk.load();
				chunkMap[chunkKey] = newChunk;
				visibleChunks.push_back(chunkKey);
			}
		}
	}
}


// TESTING 
void genCube(std::vector<GLfloat> &vertices, std::vector<GLuint> &indices) {
	vertices = {
		// Front face (z = +0.5)
		-0.5f, -0.5f,  0.5f,   0.0f,  0.0f,  1.0f,   0.0f, 0.0f,
		 0.5f, -0.5f,  0.5f,   0.0f,  0.0f,  1.0f,   1.0f, 0.0f,
		 0.5f,  0.5f,  0.5f,   0.0f,  0.0f,  1.0f,   1.0f, 1.0f,
		-0.5f,  0.5f,  0.5f,   0.0f,  0.0f,  1.0f,   0.0f, 1.0f,
		// Back face (z = -0.5)
		-0.5f, -0.5f, -0.5f,   0.0f,  0.0f, -1.0f,   1.0f, 0.0f,
		 0.5f, -0.5f, -0.5f,   0.0f,  0.0f, -1.0f,   0.0f, 0.0f,
		 0.5f,  0.5f, -0.5f,   0.0f,  0.0f, -1.0f,   0.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,   0.0f,  0.0f, -1.0f,   1.0f, 1.0f,
		// Left face (x = -0.5)
		-0.5f, -0.5f, -0.5f,  -1.0f,  0.0f,  0.0f,   0.0f, 0.0f,
		-0.5f, -0.5f,  0.5f,  -1.0f,  0.0f,  0.0f,   1.0f, 0.0f,
		-0.5f,  0.5f,  0.5f,  -1.0f,  0.0f,  0.0f,   1.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,  -1.0f,  0.0f,  0.0f,   0.0f, 1.0f,
		// Right face (x = +0.5)
		 0.5f, -0.5f, -0.5f,   1.0f,  0.0f,  0.0f,   1.0f, 0.0f,
		 0.5f, -0.5f,  0.5f,   1.0f,  0.0f,  0.0f,   0.0f, 0.0f,
		 0.5f,  0.5f,  0.5f,   1.0f,  0.0f,  0.0f,   0.0f, 1.0f,
		 0.5f,  0.5f, -0.5f,   1.0f,  0.0f,  0.0f,   1.0f, 1.0f,
		 // Top face (y = +0.5)
		 -0.5f,  0.5f,  0.5f,   0.0f,  1.0f,  0.0f,   0.0f, 0.0f,
		  0.5f,  0.5f,  0.5f,   0.0f,  1.0f,  0.0f,   1.0f, 0.0f,
		  0.5f,  0.5f, -0.5f,   0.0f,  1.0f,  0.0f,   1.0f, 1.0f,
		 -0.5f,  0.5f, -0.5f,   0.0f,  1.0f,  0.0f,   0.0f, 1.0f,
		 // Bottom face (y = -0.5)
		 -0.5f, -0.5f,  0.5f,   0.0f, -1.0f,  0.0f,   0.0f, 0.0f,
		  0.5f, -0.5f,  0.5f,   0.0f, -1.0f,  0.0f,   1.0f, 0.0f,
		  0.5f, -0.5f, -0.5f,   0.0f, -1.0f,  0.0f,   1.0f, 1.0f,
		 -0.5f, -0.5f, -0.5f,   0.0f, -1.0f,  0.0f,   0.0f, 1.0f
	};

	indices;
	indices.reserve(36);
	for (GLuint f = 0; f < 6; ++f) {
		GLuint base = f * 4;
		// two triangles per face
		indices.push_back(base + 0);
		indices.push_back(base + 1);
		indices.push_back(base + 2);
		indices.push_back(base + 0);
		indices.push_back(base + 2);
		indices.push_back(base + 3);
	}
}



// INITIALIZATION
GLFWwindow* initOpenGL()
{
	if (!glfwInit()) {		// Initialize GLFW
		std::cerr << "Failed to start GLFW" << std::endl;
		return nullptr;
	}

	GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "Procedural Terrain Generator", NULL, NULL);

	if (!window) {			// Check if window was created
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return nullptr;
	}

	glfwMakeContextCurrent(window);								// Make the window's context current

	glfwSetKeyCallback(window, KeyboardCallback);				// Set keyboard callback
	glfwSetCursorPosCallback(window, MouseMoveCallback);		// Set mouse movement callback
	glfwSetScrollCallback(window, MouseScrollCallback);			// Set mouse scroll callback
	glfwSetMouseButtonCallback(window, MouseButtonCallback);	// Set mouse button callback

	glewExperimental = GL_TRUE;									// Needed for core profile
	if (glewInit() != GLEW_OK) {								// Initialize GLEW
		std::cerr << "Failed to start GLEW" << std::endl;
		return nullptr;
	}

	// OpenGL configuration
	glViewport(0, 0, screenWidth, screenHeight);		// Set the viewport
	glEnable(GL_DEPTH_TEST);							// Enable depth testing
	glEnable(GL_CULL_FACE);								// Enable face culling
	glEnable(GL_BLEND);									// Enable blending (transparency)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);					// Set blending function (transparency)
	glClearColor(skyColor.x, skyColor.y, skyColor.z, skyColor.w);		// Set clear color (sky blue)

	glfwShowWindow(window); // Show window
	
	return window;
}