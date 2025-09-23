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
// Basic vertex shader
const char* vertexShaderSource = R"glsl(
	#version 330 core
	layout(location = 0) in vec3 aPos;    // Vertex position
	layout(location = 1) in vec3 aNormal; // Vertex normal
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
	void main() {
		// Ambient
		float ambientStrength = 0.3;
		vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
		// Diffuse
		vec3 norm = normalize(Normal);
		vec3 lightDir = normalize(lightPos - FragPos);
		float diff = max(dot(norm, lightDir), 0.0);
		vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
		// Specular
		float specularStrength = 0.5;
		vec3 viewDir = normalize(viewPos - FragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
		vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
		vec3 result = (ambient + diffuse + specular) * texture(texture1, TexCoord).rgb;
		FragColor = vec4(result, 1.0);
})glsl";



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
void CreateBufferArrayObjects(GLuint& VBO, GLuint& VAO, GLuint& EBO, const float* vertices, size_t vertexCount, const GLuint* indices, size_t indexCount); // Add Vertex Array Object to new Vertex Buffer Object using data from vertices array
void UpdateBufferArrayObjects(GLuint VBO, const float* vertices, size_t vertexCount); // Update existing Vertex Buffer Object with new data from vertices array (Add new chunks)

// SHADING
GLuint CompileShaderProgram(const char* vertexSource, const char* fragmentSource); // Compile and link vertex and fragment shaders into a shader program

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
	GLuint shaderProgram = CompileShaderProgram(vertexShaderSource, fragmentShaderSource);

	// Declare terrain data structures
	std::vector<GLfloat> terrainVertices;
	std::vector<GLuint> terrainIndices;

	GLuint terrainVBO, terrainVAO, terrainEBO; // Vertex Buffer Object, Vertex Array Object, Element Buffer Object

	// Generate terrain mesh
	generateTerrainMesh(terrainVertices, terrainIndices, perlinNoise2D);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		UpdateCamera(shaderProgram, cameraPos);


		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}



// RENDERING
void UpdateCamera(GLuint shaderProgram, glm::vec3 cameraPos) {
	glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)screenWidth / (float)screenHeight, 0.1f, 100.0f);
	glUseProgram(shaderProgram);
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
	/*
	glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // Unbind EBO
	*/
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


// CALLBACKS
void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true); // Close window on ESC key press
	if (key == GLFW_KEY_P && action == GLFW_PRESS)
		isPaused = !isPaused; // Toggle pause on P key press
}
void MouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
	// Placeholder for mouse movement handling (e.g., camera rotation)
}
void MouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	// Placeholder for mouse scroll handling (e.g., zooming)
}
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	// Placeholder for mouse button handling (e.g., selecting terrain points)
}



// TERRAIN GENERATION
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
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	// Set blending function (transparency)

	glfwShowWindow(window); // Show window
	
	return window;
}