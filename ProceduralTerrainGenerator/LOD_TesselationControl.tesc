#version 400 core
layout(vertices = 4) out;	// Pass 4 vertex per patch

const float MIN_TESS_LEVEL = 1.0;
const float MAX_TESS_LEVEL = 16.0;
const float DIST_TESS_FACTOR = 0.1; 


uniform vec3 cameraPos;		// Camera position in world space

in vec3 FragPos[];    // From vertex shader
in vec3 Normal[];
in vec2 TexCoord[];

out vec3 tcsFragPos[];
out vec3 tcsNormal[];
out vec2 tcsTexCoord[];

void main() {
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

	// Pass through vertex attributes
    tcsFragPos[gl_InvocationID] = FragPos[gl_InvocationID];
    tcsNormal[gl_InvocationID] = Normal[gl_InvocationID];
    tcsTexCoord[gl_InvocationID] = TexCoord[gl_InvocationID];

	// Compute center
	vec3 p0 = FragPos[0].xyz;
	vec3 p2 = FragPos[2].xyz;
	vec3 center = (p0 + p2) * 0.5;

	// Compute distance to camera
	float dist = distance(center, cameraPos) * DIST_TESS_FACTOR;

	// Compute tessellation level based on distance (simple linear)
	float tessLevel = clamp(MAX_TESS_LEVEL / dist, MIN_TESS_LEVEL, MAX_TESS_LEVEL);

	// Set tessellation levels for inner and outer edges
	gl_TessLevelInner[0] = tessLevel;
	gl_TessLevelInner[1] = tessLevel;
	
	gl_TessLevelOuter[0] = tessLevel;
	gl_TessLevelOuter[1] = tessLevel;
	gl_TessLevelOuter[2] = tessLevel;
	gl_TessLevelOuter[3] = tessLevel;
}

