#version 330 core
out vec4 FragColor;

in vec2 vLocal;

uniform vec3 sunColor = vec3(1.0, 0.9, 0.8);
//uniform vec3 sunsetColor = vec3(1.0, 0.25, 0.1);
uniform float innerRadius;
uniform float outerRadius;
uniform float intensity;

void main() {
	// Calculate distance from center
	float dist = length(vLocal);
	// Smoothly interpolate alpha based on distance
	float core = smoothstep(outerRadius, innerRadius*0.9, dist);
	float halo = smoothstep(outerRadius * 1.1, outerRadius, dist);
	// Final color with intensity and alpha
	float alpha = max(core, halo * 0.6) * intensity; // Remove light stacking 
	vec3 color = sunColor * intensity * alpha;
	FragColor = vec4(color, alpha);

	// Discard fragments with very low alpha to avoid unnecessary blending
	if(FragColor.a < 0.01) discard;
}