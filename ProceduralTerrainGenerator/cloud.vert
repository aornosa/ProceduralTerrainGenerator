#version 430 core
out vec2 uv;

void main()
{
	const vec2 pos[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 p = pos[gl_VertexID];

	uv = p * 0.5 + 0.5; // Map from [-1, 3] to [0, 1]

	gl_Position = vec4(p, 0.0, 1.0);	// Set vertex position
}