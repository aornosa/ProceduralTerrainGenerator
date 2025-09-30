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
}