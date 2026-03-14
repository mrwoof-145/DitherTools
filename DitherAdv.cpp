// dither.cpp
#define _CRT_SECURE_NO_WARNINGS

#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Quantize helper (0..8 bits)
inline uint8_t quantize(uint8_t value, int bits) {
    if (bits >= 8) return value;
    if (bits <= 0) return 0;
    int levels = 1 << bits;
    int q = (value * (levels - 1) + 127) / 255;
    int out = (q * 255 + (levels - 1) / 2) / (levels - 1);
    return static_cast<uint8_t>(out);
}

// 4x4 Bayer matrix
static const int bayer4[4][4] = {
    { 0,  8,  2, 10},
    {12,  4, 14,  6},
    {3, 11,  1,  9},
    {15,  7, 13,  5}
};

// Parallel Bayer ordered dither on an image buffer (w x h, channels)
void dither_bayer_parallel_buf(std::vector<uint8_t>& img, int width, int height, int channels, int bits) {
    if (bits >= 8) return;
    const int levels = 1 << bits;

#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int bx = x & 3;
            int by = y & 3;
            int threshold = bayer4[by][bx]; // 0..15
            int t = (threshold * 255) / 16; // bias 0..255
            int baseIdx = (y * width + x) * channels;
            for (int c = 0; c < channels; ++c) {
                int idx = baseIdx + c;
                int v = img[idx];
                int scaled = v * levels + t;
                int level = scaled / 256;
                if (level < 0) level = 0;
                if (level >= levels) level = levels - 1;
                int out = (level * 255 + (levels - 1) / 2) / (levels - 1);
                img[idx] = static_cast<uint8_t>(out);
            }
        }
    }
}

// Floyd–Steinberg error diffusion on buffer (sequential)
void dither_floyd_steinberg_buf(std::vector<uint8_t>& img, int width, int height, int channels, int bits) {
    if (bits >= 8) return;

    auto idx = [&](int x, int y, int c) { return (y * width + x) * channels + c; };

    std::vector<float> buffer((size_t)width * height * channels);
    for (size_t i = 0; i < buffer.size(); ++i) buffer[i] = static_cast<float>(img[i]);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int i = idx(x, y, c);
                float oldf = buffer[i];
                int oldPixel = static_cast<int>(std::round(oldf));
                uint8_t newPixel = quantize(static_cast<uint8_t>(std::max(0, std::min(255, oldPixel))), bits);
                float error = oldf - static_cast<float>(newPixel);
                img[i] = newPixel;
                if (x + 1 < width) buffer[idx(x + 1, y, c)] += error * (7.0f / 16.0f);
                if (x > 0 && y + 1 < height) buffer[idx(x - 1, y + 1, c)] += error * (3.0f / 16.0f);
                if (y + 1 < height) buffer[idx(x, y + 1, c)] += error * (5.0f / 16.0f);
                if (x + 1 < width && y + 1 < height) buffer[idx(x + 1, y + 1, c)] += error * (1.0f / 16.0f);
            }
        }
    }
}

// Pixelation + dither pipeline:
// - Given original img (w x h x channels) and res (float), compute block size B = max(1, round(1.0/res))
// - Downsample into small image by averaging each BxB block
// - Apply chosen dither on small image
// - Upscale by nearest neighbor (replicate) to original size
bool dither_with_resolution(std::vector<uint8_t>& img, int width, int height, int channels, int bits,
    const std::string& mode, double res) {
    if (res <= 0.0) return false;
    if (res >= 1.0) {
        // no pixelation, just run dither on original buffer
        if (mode == "floyd") dither_floyd_steinberg_buf(img, width, height, channels, bits);
        else dither_bayer_parallel_buf(img, width, height, channels, bits);
        return true;
    }

    // compute block size: smaller res -> larger blocks
    int B = static_cast<int>(std::round(1.0 / res));
    if (B < 1) B = 1;

    int w2 = (width + B - 1) / B;
    int h2 = (height + B - 1) / B;

    // small image buffer (w2 * h2 * channels), store averaged colors
    std::vector<uint8_t> small((size_t)w2 * h2 * channels, 0);

    for (int by = 0; by < h2; ++by) {
        for (int bx = 0; bx < w2; ++bx) {
            int x0 = bx * B;
            int y0 = by * B;
            int x1 = std::min(width, x0 + B);
            int y1 = std::min(height, y0 + B);
            int count = 0;
            std::vector<int> sum(channels, 0);
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    int base = (y * width + x) * channels;
                    for (int c = 0; c < channels; ++c) sum[c] += img[base + c];
                    ++count;
                }
            }
            int sidx = (by * w2 + bx) * channels;
            if (count == 0) count = 1;
            for (int c = 0; c < channels; ++c) {
                small[sidx + c] = static_cast<uint8_t>(sum[c] / count);
            }
        }
    }

    // apply dither on small image
    if (mode == "floyd") {
        dither_floyd_steinberg_buf(small, w2, h2, channels, bits);
    }
    else {
        dither_bayer_parallel_buf(small, w2, h2, channels, bits);
    }

    // upscale small back to original by replication
    std::vector<uint8_t> out((size_t)width * height * channels);
    for (int by = 0; by < h2; ++by) {
        for (int bx = 0; bx < w2; ++bx) {
            int sidx = (by * w2 + bx) * channels;
            int x0 = bx * B;
            int y0 = by * B;
            int x1 = std::min(width, x0 + B);
            int y1 = std::min(height, y0 + B);
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    int oidx = (y * width + x) * channels;
                    for (int c = 0; c < channels; ++c) {
                        out[oidx + c] = small[sidx + c];
                    }
                }
            }
        }
    }

    img.swap(out);
    return true;
}

// Parse --mode=... and --res=... arguments (returns empty if not present)
static std::string parse_flag_arg(int argc, char** argv, const char* prefix) {
    size_t plen = std::strlen(prefix);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], prefix, plen) == 0) {
            return std::string(argv[i] + plen);
        }
    }
    return std::string();
}

static void print_usage_and_exit(const char* prog) {
    std::cerr << "Usage: " << prog << " input.png [output.png] [bits] [threads] [--mode=bayer|floyd] [--res=<float>]\n";
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2) print_usage_and_exit(argv[0]);

    // parse flags
    std::string mode = parse_flag_arg(argc, argv, "--mode=");
    std::string resstr = parse_flag_arg(argc, argv, "--res=");

    if (!mode.empty()) {
        if (mode != "bayer" && mode != "floyd") {
            std::cerr << "Error: unknown mode '" << mode << "'. Allowed: bayer, floyd\n";
            print_usage_and_exit(argv[0]);
        }
    }
    else {
        mode = "bayer";
    }

    double res = 1.0;
    if (!resstr.empty()) {
        try {
            res = std::stod(resstr);
        }
        catch (...) {
            std::cerr << "Error: invalid --res value\n";
            print_usage_and_exit(argv[0]);
        }
        if (res <= 0.0) {
            std::cerr << "Error: --res must be > 0\n";
            print_usage_and_exit(argv[0]);
        }
    }

    // collect positional args excluding flags
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) continue;
        if (std::strncmp(argv[i], "--res=", 6) == 0) continue;
        pos.emplace_back(argv[i]);
    }

    if (pos.empty()) print_usage_and_exit(argv[0]);

    const std::string input = pos[0];
    const std::string output = (pos.size() >= 2 && !pos[1].empty()) ? pos[1] : "dithered.png";
    int bits = 5;
    if (pos.size() >= 3) bits = std::atoi(pos[2].c_str());
    int threads = 0;
    if (pos.size() >= 4) threads = std::atoi(pos[3].c_str());

    if (bits < 0) bits = 0;
    if (bits > 8) bits = 8;

    // threads argument ONLY. If provided and >0, set OpenMP threads.
    if (threads > 0) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
    }

    int width = 0, height = 0, channels = 0;
    unsigned char* data = stbi_load(input.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load input: " << input << "\n";
        return 2;
    }

    std::vector<uint8_t> img(data, data + (size_t)width * height * channels);
    stbi_image_free(data);

    if (!dither_with_resolution(img, width, height, channels, bits, mode, res)) {
        std::cerr << "Dithering failed (invalid res?)\n";
        return 4;
    }

    if (!stbi_write_png(output.c_str(), width, height, channels, img.data(), width * channels)) {
        std::cerr << "Failed to write output: " << output << "\n";
        return 3;
    }

    return 0;
}
