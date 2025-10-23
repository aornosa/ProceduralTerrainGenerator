#version 430 core

in vec2 uv;
out vec4 FragColor;


// Projection
uniform vec3 cameraPos;
uniform mat4 invView;
uniform mat4 invProjection;

// Lighting
uniform vec3 sunDir;
uniform vec3 sunColor;
uniform float phaseG = 0.76; // Henyey-Greenstein phase function parameter default

// Cloud volume
uniform sampler3D cloudTex;				// 3D noise texture for cloud density
uniform float cloudBottom = 100.0;
uniform float cloudTop = 200.0;			// Cloud layer boundaries
uniform float stepSize;					// Step size for ray marching
uniform float densityMultiplier = 1.0;	// Overall density multiplier
uniform float coverage = 0.45;			// Cloud coverage
uniform float edgeSoftness = 0.1;		// Softness of cloud layer edges
uniform float noiseScale;				// Scale of the noise texture

vec3 reconstructWorldRay(vec2 uv) {
	// Uv into NDC
	vec2 ndc = uv * 2.0 - 1.0;

	// NDC to clip space
	vec4 clip = vec4(ndc, -1.0, 1.0);

	// Back project to view space
	vec4 view = invProjection * clip;
	view /= view.w;

	// Transform to world space
	vec3 rayDir = normalize((invView * vec4(view.xyz, 0.0)).xyz);
	return normalize(rayDir);
}

float hgPhase (float cosTheta, float g) {
	float g2 = g * g;
	float denom = pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
	return (1.0 - g2) / (4.0 * 3.14159265 * denom);
}

float heightMask(float y) {
    float h = clamp((y - cloudBottom) / max(cloudTop - cloudBottom, 1e-3), 0.0, 1.0);
    float b = smoothstep(0.0, edgeSoftness, h);
    float t = 1.0 - smoothstep(1.0 - edgeSoftness, 1.0, h);
    return b * t;
}

float sampleCloudDensity(vec3 pos) {
	vec3 localPos = pos * noiseScale;
	float d = max(texture(cloudTex, fract(localPos)).r - coverage, 0.0);  // umbral de cobertura
    d *= heightMask(pos.y);
    return d; 
}


vec4 rayMarch (vec3 rO, vec3 rD, float t0, float t1) {
	vec3 color = vec3(0.0);
	float transmittance = 1.0;

	float t = t0;

	for(int i = 0; i < 128 && t < t1; i++) {
		vec3 pos = 	rO + rD * t;
		float density = sampleCloudDensity(pos);

		if(density > 0.01) // Early exit for low density
		{
			float sigma_t = density * densityMultiplier;
			float stepTransmittance = exp(-sigma_t * stepSize);

			// Shadow approximation
			float lightDensity = sampleCloudDensity(pos + sunDir * 1000.0);
			float lightTransmittance = exp(-lightDensity * densityMultiplier * 0.5);

			float phase = hgPhase(dot(rD, sunDir), phaseG);

			vec3 scattering = sunColor * lightTransmittance * phase * sigma_t;

			color += transmittance * scattering * (1.0 - stepTransmittance);
			transmittance *= stepTransmittance;

			if (transmittance < 0.001) break;
		}

		t += stepSize;
	}

	return vec4(color, 1.0 - transmittance);
}

void main() {
	vec3 rayOrigin = cameraPos;
	vec3 rayDir = reconstructWorldRay(uv);

	// Compute intersection with cloud layer
	float t0 = (cloudBottom - rayOrigin.y) / rayDir.y;
	float t1 = (cloudTop - rayOrigin.y) / rayDir.y;

	if(t0 > t1) {
		float temp = t0;
		t0 = t1;
		t1 = temp;
	}

	if (t1 < 0.0) discard; // Ray points away from clouds
	t0 = max(t0, 0.0); // Start from the camera if inside cloud layer

	vec4 cloudColor = rayMarch(rayOrigin, rayDir, t0, t1);

	FragColor = cloudColor;
}