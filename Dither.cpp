// dither.cpp
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Quantize a single channel to N bits
inline uint8_t quantize(uint8_t value, int bits) {
    int levels = 1 << bits;
    int step = 256 / levels;
    return static_cast<uint8_t>(std::round(value / step) * step);
}

// Floyd–Steinberg dithering
void dither(std::vector<uint8_t>& img, int width, int height, int channels, int bits) {
    auto idx = [&](int x, int y, int c) { return (y * width + x) * channels + c; };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int i = idx(x, y, c);
                uint8_t oldPixel = img[i];
                uint8_t newPixel = quantize(oldPixel, bits);
                int error = oldPixel - newPixel;
                img[i] = newPixel;

                if (x + 1 < width) img[idx(x + 1, y, c)] = std::clamp(img[idx(x + 1, y, c)] + error * 7 / 16, 0, 255);
                if (x > 0 && y + 1 < height) img[idx(x - 1, y + 1, c)] = std::clamp(img[idx(x - 1, y + 1, c)] + error * 3 / 16, 0, 255);
                if (y + 1 < height) img[idx(x, y + 1, c)] = std::clamp(img[idx(x, y + 1, c)] + error * 5 / 16, 0, 255);
                if (x + 1 < width && y + 1 < height) img[idx(x + 1, y + 1, c)] = std::clamp(img[idx(x + 1, y + 1, c)] + error * 1 / 16, 0, 255);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.png> <output.png> <bits>\n";
        return 1;
    }

    std::string input = argv[1];
    std::string output = argv[2];
    int bits = std::stoi(argv[3]);

    int width, height, channels;
    unsigned char* data = stbi_load(input.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load image: " << input << "\n";
        return 1;
    }

    std::vector<uint8_t> img(data, data + width * height * channels);
    stbi_image_free(data);

    dither(img, width, height, channels, bits);

    if (!stbi_write_png(output.c_str(), width, height, channels, img.data(), width * channels)) {
        std::cerr << "Failed to write image: " << output << "\n";
        return 1;
    }

    return 0;
}
