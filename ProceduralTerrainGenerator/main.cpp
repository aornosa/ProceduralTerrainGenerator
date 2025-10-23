#include <windows.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>

#include "noise.hpp"
#include "tmath.hpp"

// SHADER SOURCE PATHS
const char* vertexShaderPath = "./generic.vert";
const char* fragmentShaderPath = "./generic.frag";
const char* tesselationControlShaderPath = "./LOD_TesselationControl.tesc";
const char* tesselationEvaluationShaderPath = "./LOD_TesselationEvaluation.tese";

const char* sunVertexShaderPath = "./sun.vert";
const char* sunFragmentShaderPath = "./sun.frag";

const char* cloudVertexShaderPath = "./cloud.vert";
const char* cloudFragmentShaderPath = "./cloud.frag";


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
const int terrainGridSize = 16;								// Powers of 2. Will change terrain resolution but not size
const float terrainVertexSpacing = 64/terrainGridSize;		// Only applies to X and Z axis
const float terrainRenderResolution = 0.5;					// Render resolution (0.1 = 10% of vertices rendered, 1.0 = 100% of vertices rendered)
const float terrainHeightScale = 15.f;						// Scale height (Y axis)
const float terrainFrequency = 0.0005f;						// Frequency of the noise function
const int noiseOctaveN = 16;								// Number of noise layers
const int seed = 12345;										// IMPLEMENT RANDOM SEEDING

// SKY SETTINGS
glm::vec4 defaultSkyColor = { 0.5f, 0.7f, 1.0f, 1.0f };
glm::vec4 skyColor = defaultSkyColor;

const glm::vec3 sunDayColor = glm::vec3(1.00f, 0.9f, 0.8f);
const glm::vec3 sunDuskColor = glm::vec3(1.0, 0.25, 0.1);
glm::vec3 sunColor = sunDuskColor;

const float fogDensity = 0.001;
glm::vec3 fogColor = { 0.55f, 0.75f, 1.0f };

// DAY/NIGHT CYCLE SETTINGS
const bool enableDayNightCycle = true;
float timeOfDay = 8.0f;			// Current time of day (0.0 - 24.0)
const float timeSpeed = 0.1f;	// Speed of time progression (hours per second)


// RENDER SETTINGS
const int renderDistance = 16;			// Render distance in chunks (Default: 16)


GLuint cloudTexture;
static GLuint dummyVAO = 0;


// OBJECT DECLARATIONS
class Chunk;



// FUNCTION DECLARATIONS
GLFWwindow* initOpenGL();

// RENDERING
void UpdateCamera(GLuint shaderProgram, glm::vec3 cameraPos);
void CreateBufferArrayObjects(GLuint& VBO, GLuint& VAO, GLuint& EBO, const float* vertices, size_t vertexCount, const GLuint* indices, size_t indexCount); // Add Vertex Array Object to new Vertex Buffer Object using data from vertices array
void UpdateBufferArrayObjects(GLuint& VBO, GLuint& EBO, const float* vertices, size_t vertexCount, const GLuint* indices, size_t indexCount); // Update existing Vertex Buffer Object with new data from vertices array (Add new chunks)

// SHADING
std::string LoadShaderSource(const char* filePath); // Load shader source code from file
GLuint CompileShaderProgram(const char* vertexSource, const char* fragmentSource, const char* tesselationControlSource, const char* tesselationEvaluationSource); // Compile and link vertex and fragment shaders into a shader program
GLuint CompileShaderProgram(const char* vertexSource, const char* fragmentSource); // Overloaded function for shaders without tesselation

// DRAWING
void DrawSun(GLuint shaderProgram, glm::vec3 &sunPosition); // Draw the sun
void DrawClouds(GLuint shaderProgram, glm::vec3& sunPosition); // Draw clouds

// CALLBACKS
void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);	// Callback for keyboard input
void MouseMoveCallback(GLFWwindow* window, double xpos, double ypos);					// Callback for mouse movement input
void MouseScrollCallback(GLFWwindow* window, double xpos, double ypos);					// Callback for mouse scroll wheel input
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);			// Callback for mouse button input

GLuint generateCloudTexture3D(); // Generate 3D texture for clouds

// TERRAIN GENERATION
void generateTerrainMesh(std::vector<GLfloat> &vertices, std::vector<GLuint> &indices, double (*noise_fun)(double, double, int), glm::vec2 chunkPos);	// Generate terrain mesh using Perlin noise

// TERRAIN RENDERING
void updateVisibleChunks(std::unordered_map<long long, Chunk>& chunkMap, std::vector<long long>& visibleChunks, glm::vec3 cameraPos, int renderDistance);

// SKY RENDERING
glm::vec3 calculateSunPosition(float timeOfDay); // Calculate sun position based on time of day


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
	
	// Load and compile generic shader program
	std::string vertexShaderSource = LoadShaderSource(vertexShaderPath);
	std::string fragmentShaderSource = LoadShaderSource(fragmentShaderPath);
	std::string tesselationControlShaderSource = LoadShaderSource(tesselationControlShaderPath);
	std::string tesselationEvaluationShaderSource = LoadShaderSource(tesselationEvaluationShaderPath);
	
	GLuint shaderProgram = CompileShaderProgram(vertexShaderSource.c_str(), fragmentShaderSource.c_str(),
												tesselationControlShaderSource.c_str(), tesselationEvaluationShaderSource.c_str());


	// Load and compile sun shader program 
	std::string sunVertexShaderSource = LoadShaderSource(sunVertexShaderPath);
	std::string sunFragmentShaderSource = LoadShaderSource(sunFragmentShaderPath);

	GLuint sunShaderProgram = CompileShaderProgram(sunVertexShaderSource.c_str(), sunFragmentShaderSource.c_str());

	// Load and compile cloud shader program
	std::string cloudVertexShaderSource = LoadShaderSource(cloudVertexShaderPath);
	std::string cloudFragmentShaderSource = LoadShaderSource(cloudFragmentShaderPath);

	GLuint cloudShaderProgram = CompileShaderProgram(cloudVertexShaderSource.c_str(), cloudFragmentShaderSource.c_str());

	cloudTexture = generateCloudTexture3D();


	// Data structures
	std::unordered_map<long long, Chunk> chunkMap; // Map to store chunks by their position key
	std::vector<long long> visibleChunks; // List of currently loaded chunk keys


	// Create a simple 1x1 white texture so shader sampling is valid (CHANGE FOR TEXTURE GENERATION/SAMPLING)
	GLuint whiteTexture = 0;
	glGenTextures(1, &whiteTexture);
	glBindTexture(GL_TEXTURE_2D, whiteTexture);
	unsigned char whitePixel[3] = { 45, 150, 75 };
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, whitePixel);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);


	double lastFrameTime = glfwGetTime();

	while (!glfwWindowShouldClose(window)) {
		// Calculate delta time
		double currentFrameTime = glfwGetTime();
		double deltaTime = currentFrameTime - lastFrameTime;
		lastFrameTime = currentFrameTime;

		// Clear screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Update time of day
		if (enableDayNightCycle && !isPaused) {
			timeOfDay += timeSpeed * (float)deltaTime;
			if (timeOfDay >= 24.0f) timeOfDay -= 24.0f;
			else if (timeOfDay < 0.0f) timeOfDay += 24.0f;
		}

		glm::vec3 sunWorldPos = calculateSunPosition(timeOfDay);

		// Update graphical stuff
		UpdateCamera(shaderProgram, cameraPos);
		updateVisibleChunks(chunkMap, visibleChunks, cameraPos, renderDistance); // Update visible chunks based on camera position and render distance

		// TESTING RENDER CUBE
		glBindTexture(GL_TEXTURE_2D, whiteTexture);
		GLint texLoc = glGetUniformLocation(shaderProgram, "texture1");
		if (texLoc >= 0) glUniform1i(texLoc, 0);


		// Link shader program
		glUseProgram(shaderProgram);

		// Set fog values
		GLint fogDensityLoc = glGetUniformLocation(shaderProgram, "fogDensity");
		if (fogDensityLoc >= 0) glUniform1f(fogDensityLoc, fogDensity);

		fogColor = skyColor*0.5f;

		GLint fogColorLoc = glGetUniformLocation(shaderProgram, "fogColor");
		if (fogColorLoc >= 0) glUniform3f(fogColorLoc, fogColor.r, fogColor.g, fogColor.b);


		// Set light and view position uniforms
		GLint lightLoc = glGetUniformLocation(shaderProgram, "lightPos");
		if (lightLoc >= 0) glUniform3f(lightLoc, sunWorldPos.x, sunWorldPos.y, sunWorldPos.z);

		GLint lightColorLoc = glGetUniformLocation(shaderProgram, "lightColor");
		if (lightColorLoc >= 0) glUniform3f(lightColorLoc, sunColor.r, sunColor.g, sunColor.b);

		GLint viewLocP = glGetUniformLocation(shaderProgram, "viewPos");
		if (viewLocP >= 0) glUniform3f(viewLocP, cameraPos.x, cameraPos.y, cameraPos.z);


		// Render visible chunks
		for (long long keys : visibleChunks) {
			Chunk& chunk = chunkMap.at(keys);
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
			glDrawElements(GL_PATCHES, (GLsizei)chunk.indices.size(), GL_UNSIGNED_INT, 0);

			glBindVertexArray(0);
		}

		// Render sun
		DrawSun(sunShaderProgram, sunWorldPos);
		DrawClouds(cloudShaderProgram, sunWorldPos);

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
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

	GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
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
std::string LoadShaderSource(const char *filePath) {
	std::ifstream file(filePath, std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << filePath << std::endl;
		return "";
	}

	std::string source((std::istreambuf_iterator<char>(file)),
		std::istreambuf_iterator<char>());
	file.close();
	return source;
}
GLuint CompileShaderProgram(const char *vertexSource, const char *fragmentSource, const char *tesselationControlSource, const char *tesselationEvaluationSource) {
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

	// Tesselation Control Shader
	GLuint tesselationControlShader = glCreateShader(GL_TESS_CONTROL_SHADER);	// Create tesselation control shader
	glShaderSource(tesselationControlShader, 1, &tesselationControlSource, NULL); // Attach tesselation control shader source code
	glCompileShader(tesselationControlShader);						// Compile tesselation control shader

	glGetShaderiv(tesselationControlShader, GL_COMPILE_STATUS, &success); // Check for compilation errors
	if (!success) {
		GLchar infoLog[512];
		glGetShaderInfoLog(tesselationControlShader, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::TESSELATION_CONTROL::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// Tesselation Evaluation Shader
	GLuint tesselationEvaluationShader = glCreateShader(GL_TESS_EVALUATION_SHADER); // Create tesselation evaluation shader
	glShaderSource(tesselationEvaluationShader, 1, &tesselationEvaluationSource, NULL); // Attach tesselation evaluation shader source code
	glCompileShader(tesselationEvaluationShader);				// Compile tesselation evaluation shader

	glGetShaderiv(tesselationEvaluationShader, GL_COMPILE_STATUS, &success); // Check for compilation errors
	if (!success) {
		GLchar infoLog[512];
		glGetShaderInfoLog(tesselationEvaluationShader, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::TESSELATION_EVALUATION::COMPILATION_FAILED\n" << infoLog << std::endl;
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
	glAttachShader(shaderProgram, tesselationControlShader);
	glAttachShader(shaderProgram, tesselationEvaluationShader);
	glAttachShader(shaderProgram, fragmentShader);				// Attach fragment shader to shader program
	glLinkProgram(shaderProgram);								// Link shader program

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);	// Check for linking errors

	if (!success) {
		GLchar infoLog[512];
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}

	// Clean up (Shaders already linked in program)
	glDeleteShader(vertexShader);					// Delete vertex shader
	glDeleteShader(fragmentShader);					// Delete fragment shader
	glDeleteShader(tesselationControlShader);		// Delete tesselation control shader
	glDeleteShader(tesselationEvaluationShader);	// Delete tesselation evaluation shader

	return shaderProgram;			// Return shader program ID
}
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
	glDeleteShader(vertexShader);					// Delete vertex shader
	glDeleteShader(fragmentShader);					// Delete fragment shader

	return shaderProgram;			// Return shader program ID
} //REFACTOR TO USE FULL IMPLEMENTATION

// DRAWING
void DrawTerrain(GLuint shaderProgram, GLuint VAO, size_t indexCount) {
}
void DrawSun(GLuint shaderProgram, glm::vec3& sunWorldPos) {
	// Calculate sun color and intensity based on height
	glm::vec3 dir = glm::normalize(sunWorldPos);
	float elevation = glm::clamp(dir.y, 0.0f, 1.0f);

	// Non linear interpolation for color transition
	float t = 1.0f - elevation;              // 1 at horizon, 0 at zenith
	float curve = powf(t, 1.5f);             // intensify transition near horizon

	sunColor = glm::mix(sunDayColor, sunDuskColor, glm::clamp(curve, 0.0f, 1.0f));

	// Brighter and larger sun near horizon
	float intensity = glm::mix(0.8f, 2.0f, glm::smoothstep(0.0f, 1.0f, elevation));
	skyColor = glm::mix(glm::vec4(defaultSkyColor.r, defaultSkyColor.g, defaultSkyColor.b, 1.0f), glm::vec4(0.015f, 0.015f, 0.075f, 1.0f), curve);
	glClearColor(skyColor.r, skyColor.g, skyColor.b, skyColor.a);

	// Inner and outer radius of sun glow
	float innerRadius = glm::mix(0.62f, 0.28f, elevation); // slight diffusion even at zenith
	float outerRadius = glm::mix(0.82f, 0.46f, elevation); // more diffusion near horizon


	if (sunWorldPos.y <= 0.f) return;	// Do not draw sun if below horizon

	static GLuint sunVAO = 0, sunVBO = 0;
	if (sunVAO == 0) {
		float quad[] = {
			-1.0f,  1.0f, 
			-1.0f, -1.0f, 
			 1.0f, -1.0f, 
			 1.0f,  1.0f,
		};
		glGenVertexArrays(1, &sunVAO);
		glGenBuffers(1, &sunVBO);

		glBindVertexArray(sunVAO);
		glBindBuffer(GL_ARRAY_BUFFER, sunVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

		glBindVertexArray(0);
	}

	// Project sun to NDC space
	glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)screenWidth / (float)screenHeight, 0.1f, 30000.0f);
	glm::vec4 clip = projection * view * glm::vec4(sunWorldPos, 1.0f);

	if (clip.w <= 0.) return; // Sun is behind camera
	clip /= clip.w; // Perspective divide

	if (clip.z < -1.0f || clip.z > 1.0f) return; // Sun is outside clip space

	glm::vec2 sunNDC = glm::vec2(clip.x, clip.y);

	const float sunRadiusPixels = 50.0f; // Sun radius in pixels
	float hx = sunRadiusPixels / (screenWidth / 2.0f); // Horizontal radius in NDC
	float hy = sunRadiusPixels / (screenHeight / 2.0f); // Vertical radius in NDC

	// Link shader program
	glUseProgram(shaderProgram);

	// Input uniforms
	GLint sunPosLoc = glGetUniformLocation(shaderProgram, "sunNDC");
	if (sunPosLoc >= 0) glUniform2f(sunPosLoc, sunNDC.x, sunNDC.y);

	GLint sunRadiusLoc = glGetUniformLocation(shaderProgram, "sunSize");
	if (sunRadiusLoc >= 0) glUniform2f(sunRadiusLoc, hx, hy);


	GLint sunColorLoc = glGetUniformLocation(shaderProgram, "sunColor");
	if (sunColorLoc >= 0) glUniform3fv(sunColorLoc, 1, glm::value_ptr(sunColor));

	GLint innerRadiusLoc = glGetUniformLocation(shaderProgram, "innerRadius");
	if (innerRadiusLoc >= 0) glUniform1f(innerRadiusLoc, innerRadius);

	GLint outerRadiusLoc = glGetUniformLocation(shaderProgram, "outerRadius");
	if (outerRadiusLoc >= 0) glUniform1f(outerRadiusLoc, outerRadius);

	GLint intensityLoc = glGetUniformLocation(shaderProgram, "intensity");
	if (intensityLoc >= 0) glUniform1f(intensityLoc, intensity);

	// Draw sun quad
	GLboolean prevDepthMask;
	glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);

	glDepthMask(GL_FALSE); // Disable depth writing
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glBindVertexArray(sunVAO);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(prevDepthMask); // Restore previous depth writing state
}
void DrawClouds(GLuint shaderProgram, glm::vec3& sunWorldPos) {
	// Get view and projection matrices
	glm::mat4 v = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

	glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)screenWidth / (float)screenHeight, 0.1f, 3000.0f);

	glm::vec3 sunDir = glm::normalize(sunWorldPos);

	
	if (dummyVAO == 0) {
		glGenVertexArrays(1, &dummyVAO);
	}
	glBindVertexArray(dummyVAO);

	glUseProgram(shaderProgram);
	glUniform3fv(glGetUniformLocation(shaderProgram, "cameraPos"), 1, glm::value_ptr(cameraPos));
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "invView"), 1, GL_FALSE, glm::value_ptr(glm::inverse(v)));
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "invProjection"), 1, GL_FALSE, glm::value_ptr(glm::inverse(projection)));

	glUniform3f(glGetUniformLocation(shaderProgram, "sunDir"), sunDir.x, sunDir.y, sunDir.z);
	glUniform3f(glGetUniformLocation(shaderProgram, "sunColor"), 1.0f, 0.95f, 0.85f);
	
	glUniform1f(glGetUniformLocation(shaderProgram, "cloudBottom"), 10000.0f);
	glUniform1f(glGetUniformLocation(shaderProgram, "cloudTop"), 20000.0f);
	glUniform1f(glGetUniformLocation(shaderProgram, "stepSize"), 100.0f);
	glUniform1f(glGetUniformLocation(shaderProgram, "densityMultiplier"), 0.1f);
	glUniform1f(glGetUniformLocation(shaderProgram, "coverage"), 0.45f);
	glUniform1f(glGetUniformLocation(shaderProgram, "edgeSoftness"), 0.0001f);
	glUniform1f(glGetUniformLocation(shaderProgram, "noiseScale"), 0.00001f);
	glUniform1f(glGetUniformLocation(shaderProgram, "phaseG"), 0.76f);


	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, cloudTexture);
	glUniform1i(glGetUniformLocation(shaderProgram, "cloudTex"), 0);
	
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// --- Draw fullscreen triangle ---
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	// --- Optional: unbind VAO ---
	glBindVertexArray(0);
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

GLuint generateCloudTexture3D() {
	static GLuint textureID = 0;
	if (textureID != 0) return textureID;

	// Generate 3D texture
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_3D, textureID);

	const int size = 16;
	std::vector<float> data(size * size * size);

	// Fill texture with fractal voronoi noise
	for (int z = 0; z < size; ++z) {
		for (int y = 0; y < size; ++y) {
			for (int x = 0; x < size; ++x) {
				double nx = (double)x / (double)size;
				double ny = (double)y / (double)size;
				double nz = (double)z / (double)size;
				data[x + y * size + z * size * size] = (float)fractalNoise3D(voronoiNoise3D, nx, ny, nz, 4, seed, 2.5, 0.5, 2.); // Scale coordinates for more detail
				std::cout << "Generating cloud texture x slice " << x + 1 << " / " << size << "\t " << y + 1 << " / " << size << "\t " << z + 1 << "/" << size << std::flush << "\r";
			}
		}
	}
	// Upload texture data
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, size, size, size, 0, GL_RED, GL_FLOAT, data.data());

	// Set texture parameters
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Unbind and return texture
	glBindTexture(GL_TEXTURE_3D, 0);
	return textureID;
}

// TERRAIN GENERATION
void generateTerrainMesh(std::vector<GLfloat>& vertices, std::vector<GLuint> &indices, double (*noise_fun)(double, double, int), glm::vec2 chunkPos) {
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
			double h = pow(fractalNoise2D(perlinNoise2D, worldX * terrainFrequency, worldZ * terrainFrequency, noiseOctaveN, seed) * terrainHeightScale,4); // Scale using amplitude and frequency as noise_fun(worldX*freq, worldZ*freq)*amplitude;
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

	// Build indices for patches
	/*
		The patches are built getting a square of 4 vertices from the grid.
		All patches will have a shape like this:

		(z, x)--(z+1,x+1)
		   |         |
		   |         |
		   |         |
		   |         |
		   |         |
		(z+1,x)--(z, x+1)
	*/
	for (int z = 0; z < terrainGridSize-1; ++z) {
		for (int x = 0; x < terrainGridSize-1; ++x) {
			GLuint topLeft = z * terrainGridSize + x;
			GLuint topRight = topLeft + 1;
			GLuint bottomLeft = (z + 1) * terrainGridSize + x;
			GLuint bottomRight = bottomLeft + 1;

			indices.push_back(topLeft);
			indices.push_back(bottomLeft);
			indices.push_back(bottomRight);
			indices.push_back(topRight);
		}
	}
}

// TERRAIN RENDERING
void updateVisibleChunks(std::unordered_map<long long, Chunk>& chunkMap, std::vector<long long>& visibleChunks, glm::vec3 cameraPos, int renderDistance) {
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

glm::vec3 calculateSunPosition(float timeOfDay) {
	// Assuming timeOfDay is in range [0, 24] representing hours
	float theta = ((timeOfDay-8.0) / 24.0f) * 2.0f * PI; // Convert time to angle in radians (make sunrise at 8pm)
	// Calculate sun position in sky (simple circular path)
	float zBias = 0.25f;
	glm::vec3 dir = glm::normalize(glm::vec3(cos(theta), sin(theta), zBias));
	const float radius = 1500.0f; // far away
	return dir * radius;
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

	glPatchParameteri(GL_PATCH_VERTICES, 4); // Set number of vertices per patch (tessellation)
	//glDrawArrays(GL_PATCHES, 0, 4); // Draw patches (tessellation)

	glfwShowWindow(window); // Show window
	
	return window;
}