#version 330 core
layout(location = 0) in vec3 aPos;

uniform vec2 sunNDC;	// Sun direction in Normalized Device Coordinates
uniform vec2 sunSize;	// Sun size in NDC

out vec2 vLocal;


void main()
{
	vLocal = aPos.xy; 
	vec2 pos = sunNDC + aPos.xy * sunSize * 0.5;
	gl_Position = vec4(pos, 0.0, 1.0);
}