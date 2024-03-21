#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image.h"
#include "lib/stb_image_write.h"

// COLOR_CHANNELS = 0 means we retain the original number of color channels.
const uint8_t COLOR_CHANNELS = 0;
const std::string OUTPUT_DIR = "data/output/par/";

#ifdef T4
    const int THREADS = 4;
#endif
#ifdef T8
    const int THREADS = 8;
#endif
#ifdef T16
    const int THREADS = 16;
#endif
#ifdef T32
    const int THREADS = 32;
#endif
#ifdef T64
    const int THREADS = 64;
#endif

/**
 * @brief A struct representing the dimensions of an image.
 * 
 * @param width The width of the image.
 * @param height The height of the image.
 * @param channels The number of channels in the image.
 */
struct Dim {
    int width;
    int height;
    int channels;
}; 

/**
 * @brief Checks a condition and prints an error message if the condition is false.
 * 
 * @param condition The condition to check.
 * @param message The error message to print.
 */
void check_and_print_error(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << "\n";
        exit(1);
    }
}

/**
 * @brief Computes the index of a pixel in the image.
 * 
 * @param x The x coordinate of the pixel.
 * @param y The y coordinate of the pixel.
 * @param channel The color channel of the pixel.
 * @param width The width of the image.
 * @param channels The number of channels in the image.
 * @return size_t The index of the pixel in the image.
 */
size_t index(size_t x, size_t y, int channel, int width, int channels) {
    return y * width * channels + channel + x * channels;
}

/**
 * @brief Clamps a value between a minimum and maximum value.
 * 
 * @param value The value to clamp.
 * @param min The minimum value.
 * @param max The maximum value.
 * @return size_t The clamped value.
 */
size_t clamp(size_t value, size_t min, size_t max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Computes the Sobel operator on the whole image for a specific color channel. When computing
 * the operator, if a pixel falls outside of the image, the value of the nearest pixel is used.
 * 
 * @param image The image to compute the Sobel operator on.
 * @param dimensions The dimensions of the image.
 * @param channel The color channel to compute the Sobel operator on. 
 */
uint8_t* sobel(uint8_t* image, Dim dimensions, int channel) {
    uint8_t* energy = new uint8_t[dimensions.width * dimensions.height];
    
    int width = dimensions.width;
    int height = dimensions.height;
    int channels = dimensions.channels;
    
    for (int i = 0; i < dimensions.width * dimensions.height; i++) {
        energy[i] = 0;
    }

    #pragma omp parallel for
    for (int y = 0; y < dimensions.height; y++) {
        for (int x = 0; x < dimensions.width; x++) {

            // Compute the indices for the Gx part of the Sobel operator.
            size_t index1 = index(
                clamp(x - 1, 0, width - 1), clamp(y - 1, 0, height - 1), channel, width, channels
            );
            size_t index2 = index(
                x, clamp(y - 1, 0, height - 1), channel, width, channels
            );
            size_t index3 = index(
                clamp(x + 1, 0, width - 1), clamp(y - 1, 0, height - 1), channel, width, channels
            );
            size_t index4 = index(
                clamp(x - 1, 0, width - 1), y, channel, width, channels
            );
            size_t index5 = index(
                x, y, channel, width, channels
            );
            size_t index6 = index(
                clamp(x + 1, 0, width - 1), y, channel, width, channels
            );

            int Gx = -image[index1] - 2 * image[index2] - image[index3] + image[index4] + 2 * image[index5] + image[index6];

            // Compute the indices for the Gy part of the Sobel operator.
            index1 = index(
                clamp(x - 1, 0, width - 1), clamp(y - 1, 0, height - 1), channel, width, channels
            );
            index2 = index(
                clamp(x - 1, 0, width - 1), y, channel, width, channels
            );
            index3 = index(
                clamp(x - 1, 0, width - 1), clamp(y + 1, 0, height - 1), channel, width, channels
            );
            index4 = index(
                clamp(x + 1, 0, width - 1), clamp(y - 1, 0, height - 1), channel, width, channels
            );
            index5 = index(
                clamp(x + 1, 0, width - 1), y, channel, width, channels
            );
            index6 = index(
                clamp(x + 1, 0, width - 1), clamp(y + 1, 0, height - 1), channel, width, channels
            );

            int Gy = image[index1] + 2 * image[index2] + image[index3] - image[index4] - 2 * image[index5] - image[index6];

            energy[y * dimensions.width + x] = sqrt(Gx * Gx + Gy * Gy);
        }
    }
    return energy;
}

/**
 * @brief Computes the energy of the image by calculating the Sobel operator for each color
 * channel, averaging the results.
 * 
 * @param image The image to compute the energy of.
 * @param dimensions The dimensions of the image.
 */
uint8_t* sobel(uint8_t* image, Dim dimensions) {
    int width = dimensions.width;
    int height = dimensions.height;
    int channels = dimensions.channels;
    uint8_t* energy = new uint8_t[width * height];
    uint8_t** channel_energies = new uint8_t*[channels];

    for (int c = 0; c < channels; c++) {
        channel_energies[c] = sobel(image, dimensions, c);
    }

    for (int i = 0; i < width * height; i++) {
        int sum = 0;
        for (int c = 0; c < channels; c++) {
            sum += channel_energies[c][i];
        }
        energy[i] = sum / channels;
    }

    for (int c = 0; c < channels; c++) {
        delete[] channel_energies[c];
    }
    delete[] channel_energies;

    return energy;
}

/**
 * @brief Finds the minimum energy seam in the image. It first computes the dynamic programming table
 * and then finds the minimum energy seam by selecting the pixel with the lowest energy in the last row.
 * From there, it backtracks by selecting the index with the minimum value as the next step of the seam.
 * 
 * @param energy The energy of the image.
 * @param dimensions The dimensions of the image.
 * @return std::vector<size_t> The minimum energy seam.
 */
std::vector<size_t> find_seam(uint8_t* energy, Dim dimensions) {
    int width = dimensions.width;
    int height = dimensions.height;
    
    uint32_t* dp = new uint32_t[width * height];
    
    for (int i = 0; i < width; i++) {
        dp[i] = energy[i];
    }

    // Compute the dynamic programming table.
    for (int y = 1; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Compute the indices of the elements to the left and down, center and down, and right and down.
            if (x == 0) {
                size_t index = y * width + x;
                size_t index_center = (y - 1) * width + x;
                size_t index_right = (y - 1) * width + x + 1;
                dp[index] = energy[index] + std::min(dp[index_center], dp[index_right]);
            } else if (x == width - 1) {
                size_t index = y * width + x;
                size_t index_center = (y - 1) * width + x;
                size_t index_left = (y - 1) * width + x - 1;
                dp[index] = energy[index] + std::min(dp[index_center], dp[index_left]);
            } else {
                size_t index = y * width + x;
                size_t index_center = (y - 1) * width + x;
                size_t index_left = (y - 1) * width + x - 1;
                size_t index_right = (y - 1) * width + x + 1;
                dp[index] = energy[index] + std::min({dp[index_center], dp[index_left], dp[index_right]});
            }
        }
    }

    std::vector<size_t> seam;

    // Find the minimum energy seam by selecting the pixel with the lowest energy in the last row.
    int min_index = 0;
    for (int j = 0; j < width; j++) {
        if (dp[(height - 1) * width + j] < dp[(height - 1) * width + min_index]) {
            min_index = j;
        }
    }
    seam.push_back(min_index);
    
    // For all other rows, backtrack by selecting the index with the minimum value as the next step of the seam.
    for (int y = height - 2; y > 0; y--) {
        int x_actual = seam.back();
        int x_center = x_actual;
        int x_left = x_actual - 1;
        int x_right = x_actual + 1;

        if (x_left > 0 && dp[(y - 1) * width + x_left] < dp[(y - 1) * width + x_center]) {
            x_actual = x_left;
        } else if (x_right < width - 1 && dp[(y - 1) * width + x_right] < dp[(y - 1) * width + x_center]) {
            x_actual = x_right;
        }

        seam.push_back(x_actual);
    }

    delete[] dp;
    return seam;
}

/**
 * @brief Removes a seam from the image by copying all pixels except the ones in the seam. 
 * 
 * @param image The image to remove the seam from.
 * @param dimensions The dimensions of the image.
 * @param seam The seam to remove.
 * @return uint8_t* The new image with the seam removed.
 */
uint8_t* remove_seam(uint8_t* image, Dim dimensions, std::vector<size_t> seam) {
    int new_width = dimensions.width - 1;
    int height = dimensions.height;
    int channels = dimensions.channels;

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        int old_index = y * dimensions.width;
        int new_index = y * new_width;

        // Shift all pixels except the ones in the seam.
        for (int x = 0; x < dimensions.width; x++) {
            if (x != seam[y]) {
                for (int c = 0; c < channels; c++) {
                    image[new_index * channels + c] = image[old_index * channels + c];
                }
                new_index++;
            }
            old_index++;
        }
    }

    // Resize the image
    image = (uint8_t*)realloc(image, new_width * height * channels * sizeof(uint8_t));

    return image;
}

/**
 * @brief Removes a number of seams from the image.
 * 
 * @param image The image to remove the seams from.
 * @param dimensions The dimensions of the image.
 * @param to_remove The number of seams to remove.
 * @return uint8_t* The new image with the seams removed.
 */
uint8_t* carve_seams(uint8_t* image, Dim dimensions, uint16_t to_remove) {
    int width = dimensions.width;

    for (int i = 0; i < to_remove; i++) {
        // Due to the changes in the image, the energy of the image needs to be recomputed.

        auto start = std::chrono::high_resolution_clock::now();

        uint8_t* energy = sobel(image, dimensions);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Energy " << i + 1 << ": " << elapsed.count() << "s\n";

        std::vector<size_t> seam = find_seam(energy, dimensions);

        start = std::chrono::high_resolution_clock::now();

        image = remove_seam(image, dimensions, seam);

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        std::cout << "Seam " << i + 1 << ": " << elapsed.count() << "s\n";

        delete[] energy;
        dimensions.width--;
    }

    return image;
}

/**
 * @brief The entry point of the program.
 * 
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 */
int main(int argc, char* argv[]) {
    check_and_print_error(argc == 3, "Invalid number of arguments. Usage: ./scarving <image_path> <new_width>");
    
    #if defined(T4) || defined(T8) || defined(T16) || defined(T32) || defined(T64)
        omp_set_num_threads(THREADS);
    #endif

    std::string input_image_path = argv[1];
    std::string output_image_path = OUTPUT_DIR + input_image_path.substr(input_image_path.find_last_of("/\\") + 1);
    uint16_t to_remove = std::stoi(argv[2]);

    int width, height, channels;
    uint8_t* image = stbi_load(input_image_path.c_str(), &width, &height, &channels, COLOR_CHANNELS);

    check_and_print_error(image != nullptr, "Failed to load image");

    uint8_t* energy = sobel(image, {width, height, channels});
    stbi_write_png((OUTPUT_DIR + "energy.png").c_str(), width, height, 1, energy, width);

    image = carve_seams(image, {width, height, channels}, to_remove);
    stbi_write_png(output_image_path.c_str(), width - to_remove, height, channels, image, (width - to_remove) * channels);

    delete[] energy;
    delete[] image;

    return 0;
}