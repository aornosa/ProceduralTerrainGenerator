#include "texture_loader.hpp"


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


unsigned char* loadTexture(const char* filepath, int& width, int& height, int& channels) {
	unsigned char* data = stbi_load(filepath, &width, &height, &channels, 0);
	if (!data) {
		std::cerr << "Failed to load texture: " << filepath << std::endl;
	}
	return data;
}