#version 400 core
layout(quads, equal_spacing, ccw) in;

// Input per-vertex data (from vertex shader)
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// LOD parameters
// Receive per-vertex outputs from TCS
in vec3 FragPos[];
in vec3 Normal[];
in vec2 TexCoord[];

// Pass to fragment shader
out vec3 tesFragPos;
out vec3 tesNormal;
out vec2 tesTexCoord;

void main() {
    // Tessellation coordinates (u, v) in [0,1]
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    // Get the patch vertices (assumes 4 vertices per patch)
    vec3 p0 = FragPos[0];
    vec3 p1 = FragPos[1];
    vec3 p2 = FragPos[2];
    vec3 p3 = FragPos[3];

    vec3 n0 = Normal[0];
    vec3 n1 = Normal[1];
    vec3 n2 = Normal[2];
    vec3 n3 = Normal[3];

    vec2 t0 = TexCoord[0];
    vec2 t1 = TexCoord[1];
    vec2 t2 = TexCoord[2];
    vec2 t3 = TexCoord[3];

    // Bilinear interpolation across quad
    vec3 pos = mix(mix(p0, p1, u), mix(p3, p2, u), v);
    vec3 norm = normalize(mix(mix(n0, n1, u), mix(n3, n2, u), v));
    vec2 tex = mix(mix(t0, t1, u), mix(t3, t2, u), v);

    tesFragPos = pos;
    tesNormal = norm;
    tesTexCoord = tex;

    gl_Position = projection * view * vec4(pos, 1.0);
}
