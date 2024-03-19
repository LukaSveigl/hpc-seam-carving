/**
 * @file scarving.cpp
 * @brief This program performs the seam carving algorithm on an image to reduce its size.
 */

#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

const uint8_t COLOR_CHANNELS = 3;

void print_error(const std::string& message);

std::vector<size_t> find_seam(uint8_t* energy, int width, int height);
uint8_t* visualize_seams(uint8_t* image, int width, int height, int target_width, int target_height);
uint8_t* sobel(uint8_t* image, int width, int height);
uint8_t* sobel(uint8_t* image, int width, int height, int channel);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_error("Invalid number of arguments");
    }

    std::string image_path = argv[1];

    int width, height, channels;
    uint8_t* image = stbi_load(image_path.c_str(), &width, &height, &channels, COLOR_CHANNELS);

    if (image == nullptr) {
        print_error("Failed to load image");
    }

    uint8_t* energy = sobel(image, width, height);

    stbi_write_png("data/output/energy.png", width, height, 1, energy, width);

    std::cout << "Image width: " << width << "\n";
    std::cout << "Image height: " << height << "\n";

    std::vector<size_t> seam = find_seam(energy, width, height);

    for (size_t i = 0; i < seam.size(); i++) {
        // As i represents the y coordinate, the x coordinate is the value of the seam at index i.
        int index = i * width + seam[i];

        image[index * COLOR_CHANNELS] = 255;
        image[index * COLOR_CHANNELS + 1] = 0;
        image[index * COLOR_CHANNELS + 2] = 0;
    }

    stbi_write_png("data/output/seam.png", width, height, COLOR_CHANNELS, image, width * COLOR_CHANNELS);

    uint8_t* visualized_seams = visualize_seams(image, width, height, width - 100, height);

    stbi_write_png("data/output/visualized_seams.png", width, height, COLOR_CHANNELS, visualized_seams, width * COLOR_CHANNELS);

    return 0;
}

// SEAM CARVING FUNCTIONS


/**
 * @brief Visualizes the seams that would be removed from the image.
 * 
 * @param image The image to visualize the seams on.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param target_width The target width of the image.
 * @param target_height The target height of the image.
 * @return uint8_t* The image with the seams visualized. 
 */
uint8_t* visualize_seams(uint8_t* image, int width, int height, int target_width, int target_height) {
    uint8_t* energy = sobel(image, width, height);

    for (int i = 0; i < width - target_width; i++) {
        std::vector<size_t> seam = find_seam(energy, width, height);

        for (size_t j = 0; j < seam.size(); j++) {
            int index = j * width + seam[j];

            image[index * COLOR_CHANNELS] = 255;
            image[index * COLOR_CHANNELS + 1] = 0;
            image[index * COLOR_CHANNELS + 2] = 0;
        }

        width--;
    }

    return image;
}


std::vector<size_t> find_seam(uint8_t* energy, int width, int height) {
    std::vector<size_t> seam;

    int* dp = new int[width * height];

    // Initialize the first row of the dp array with the energy values.
    for (int i = 0; i < width; i++) {
        dp[i] = energy[i];
    }

    // Calculate the minimum energy seam for each pixel in the image.
    for (int y = 1; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;

            dp[index] = energy[index] + dp[(y - 1) * width + x];

            if (x > 0) {
                dp[index] = std::min(dp[index], energy[index] + dp[(y - 1) * width + x - 1]);
            }

            if (x < width - 1) {
                dp[index] = std::min(dp[index], energy[index] + dp[(y - 1) * width + x + 1]);
            }
        }
    }

    // Find the minimum energy seam by selecting the pixel with the lowest energy in the last row.
    int min_index = 0;
    for (int i = 0; i < width; i++) {
        if (dp[(height - 1) * width + i] < dp[(height - 1) * width + min_index]) {
            min_index = i;
        }
    }

    seam.push_back(min_index);

    // Trace back the minimum energy seam from the last row to the first row.
    for (int y = height - 1; y > 0; y--) {
        int index = y * width + min_index;

        if (min_index > 0 && dp[(y - 1) * width + min_index - 1] < dp[index]) {
            min_index--;
        } else if (min_index < width - 1 && dp[(y - 1) * width + min_index + 1] < dp[index]) {
            min_index++;
        }

        seam.push_back(min_index);
    }

    delete[] dp;

    return seam;
}

/**
 * @brief Applies the Sobel operator to the entire image by applying it to seperate color channels and
 * averaging the results.
 * 
 * @param image The image to apply the Sobel operator to.
 * @param width The width of the image.
 * @param height The height of the image.
 * @return uint8_t* The image with the Sobel operator applied to it.
 */
uint8_t* sobel(uint8_t *image, int width, int height) {
    uint8_t* energy = new uint8_t[width * height];

    uint8_t* red = sobel(image, width, height, 0);
    uint8_t* green = sobel(image, width, height, 1);
    uint8_t* blue = sobel(image, width, height, 2);

    for (int i = 0; i < width * height; i++) {
        energy[i] = (red[i] + green[i] + blue[i]) / 3;
    }

    delete[] red;
    delete[] green;
    delete[] blue;

    return energy;
}

/**
 * @brief Applies the Sobel operator to the specific color channel of the image.
 * 
 * @param image The image to apply the Sobel operator to.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param channel The color channel to apply the Sobel operator to.
 * @return uint8_t* The image with the Sobel operator applied to the specific color channel. 
 */
uint8_t* sobel(uint8_t* image, int width, int height, int channel) {
    uint8_t* energy = new uint8_t[width * height];

    for (int i = 0; i < width * height; i++) {
        energy[i] = 0;
    }

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int dx = 0;
            int dy = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int index = (y + i) * width + (x + j);
                    int weight = i * i + j * j;

                    dx += image[index * COLOR_CHANNELS + channel] * j * weight;
                    dy += image[index * COLOR_CHANNELS + channel] * i * weight;
                }
            }

            energy[y * width + x] = sqrt(dx * dx + dy * dy);
        }
    }

    return energy;
}

// MISC UTILS

/**
 * @brief Prints an error message and exits the program.
 * 
 * @param message The error message to print.
 */
void print_error(const std::string& message) {
    std::cerr << "Error: " << message << "\n";
    exit(1);
}