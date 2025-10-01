# version 400 core
layout(vertices = 4) out;	// Pass 4 vertex per patch

const float MIN_TESS_LEVEL = 1.0;
const float MAX_TESS_LEVEL = 64.0;


uniform vec3 cameraPos;		// Camera position in world space

void main() {
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

	// Compute center
	vec3 p0 = gl_in[0].gl_Position.xyz;
	vec3 p2 = gl_in[2].gl_Position.xyz;
	vec3 center = (p0 + p2) * 0.5;

	// Compute distance to camera
	float dist = distance(center, cameraPos);

	// Compute tessellation level based on distance (simple linear)
	float tessLevel = clamp(MAX_TESS_LEVEL / dist, MIN_TESS_LEVEL, MAX_TESS_LEVEL);

	// Set tessellation levels for inner and outer edges
	gl_TessLevelInner[0] = tessLevel;
	gl_TessLevelInner[1] = tessLevel;
	
	// For quads, we have 4 outer levels
	gl_TessLevelOuter[0] = tessLevel;
	gl_TessLevelOuter[1] = tessLevel;
	gl_TessLevelOuter[2] = tessLevel;
	gl_TessLevelOuter[3] = tessLevel;
}

