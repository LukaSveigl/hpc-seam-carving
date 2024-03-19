#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image.h"
#include "lib/stb_image_write.h"

// COLOR_CHANNELS = 0 means we retain the original number of color channels.
const uint8_t COLOR_CHANNELS = 0;
const std::string OUTPUT_DIR = "data/output/seq/";

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
 * @brief Computes the Sobel operator on the whole image for a specific color channel. When computing
 * the operator, if a pixel falls outside of the image, the value of the nearest pixel is used.
 * 
 * @param image The image to compute the Sobel operator on.
 * @param dimensions The dimensions of the image.
 * @param channel The color channel to compute the Sobel operator on. 
 */
uint8_t* sobel(uint8_t* image, Dim dimensions, int channel) {
    uint8_t* energy = new uint8_t[dimensions.width * dimensions.height];
    for (int i = 0; i < dimensions.width * dimensions.height; i++) {
        energy[i] = 0;
    }
    for (int y = 1; y < dimensions.height - 1; y++) {
        for (int x = 1; x < dimensions.width - 1; x++) {
            size_t index1 = index(x - 1, y - 1, channel, dimensions.width, dimensions.channels);
            size_t index2 = index(x, y - 1, channel, dimensions.width, dimensions.channels);
            size_t index3 = index(x + 1, y - 1, channel, dimensions.width, dimensions.channels);
            size_t index4 = index(x - 1, y + 1, channel, dimensions.width, dimensions.channels);
            size_t index5 = index(x, y + 1, channel, dimensions.width, dimensions.channels);
            size_t index6 = index(x + 1, y + 1, channel, dimensions.width, dimensions.channels);

            int Gx = -image[index1] - 2 * image[index2] - image[index3] + image[index4] + 2 * image[index5] + image[index6];

            index1 = index(x - 1, y - 1, channel, dimensions.width, dimensions.channels);
            index2 = index(x - 1, y, channel, dimensions.width, dimensions.channels);
            index3 = index(x - 1, y + 1, channel, dimensions.width, dimensions.channels);
            index4 = index(x + 1, y - 1, channel, dimensions.width, dimensions.channels);
            index5 = index(x + 1, y, channel, dimensions.width, dimensions.channels);
            index6 = index(x + 1, y + 1, channel, dimensions.width, dimensions.channels);

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

std::vector<std::vector<size_t>> find_seams(uint8_t* energy, Dim dimensions, uint16_t to_remove) {
    int width = dimensions.width;
    int height = dimensions.height;
    
    stbi_write_png((OUTPUT_DIR + "tmp-energy.png").c_str(), width, height, 1, energy, width);

    std::vector<std::vector<size_t>> seams;

    for (int i = 0; i < to_remove; i++) {
        seams.push_back(std::vector<size_t>());
    }

    std::cout << "Reinitialized seams\n";
    
    uint8_t* dp = new uint8_t[width * height];
    
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

    // Find the n seams with the lowest energy.
    for (int i = 0; i < to_remove; i++) {
        std::vector<size_t> seam;
        int min_index = 0;
        for (int j = 0; j < width; j++) {

            bool found = false;

            // Check if the begginning of the seam has not been used in a previous seam.
            for (int k = 0; k < seams.size(); k++) {
                if (seams[k][0] == j || seams[k][0]) {
                    found = true;
                }
                if (seams[k][0] == min_index) {
                    min_index++;
                }
            }

            if (found) {
                //std::cout << "Skipping " << j << "\n";
                continue;
            }

            //std::cout << "Comparing " << (int)dp[(height - 1) * width + j] << " and " << (int) dp[(height - 1) * width + min_index] << "\n";

            if (dp[(height - 1) * width + j] < dp[(height - 1) * width + min_index]) {
                min_index = j;
            }
        }
        seam.push_back(min_index);
        
        for (int y = height - 2; y >= 0; y--) {
            int x = seam.back();
            if (x > 0 && dp[y * width + x - 1] < dp[y * width + x]) {
                x--;
            }
            else if (x < width - 1 && dp[y * width + x + 1] < dp[y * width + x]) {
                x++;
            }
            seam.push_back(x);
        }

        // Print first 20 elements of a seam.
        std::cout << "Seam " << i << " before reverse: ";
        for (int j = 0; j < 20; j++) {
            std::cout << seam[j] << " ";
        }

        std::cout << "  ||  ";

        // Print last 20 elements of a seam.
        for (int j = seam.size() - 20; j < seam.size(); j++) {
            std::cout << seam[j] << " ";
        }

        std::cout << "\n";

        std::fflush(stdout);

        std::reverse(seam.begin(), seam.end());

        // Print first 20 elements of a seam.
        std::cout << "Seam " << i << " after reverse: ";
        for (int j = 0; j < 20; j++) {
            std::cout << seam[j] << " ";
        }

        std::cout << "  ||  ";

        // Print last 20 elements of a seam.
        for (int j = seam.size() - 20; j < seam.size(); j++) {
            std::cout << seam[j] << " ";
        }

        std::cout << "\n\n\n\n";
        std::fflush(stdout);

        seams[i] = std::vector<size_t>(seam.begin(), seam.end());
    }

    delete[] dp;

    return seams;
}

void visualize_seams(uint8_t* image, Dim dimensions, std::vector<std::vector<size_t>> seams) {
    int width = dimensions.width;
    int height = dimensions.height;
    int channels = dimensions.channels;

    for (int i = 0; i < seams.size(); i++) {
        for (int j = 0; j < seams[i].size(); j++) {
            size_t ind = index(seams[i][j], height - 1 - j, 0, width, channels);
            image[ind] = 255;
            image[ind + 1] = 0;
            image[ind + 2] = 0;
        }
    }

    stbi_write_png((OUTPUT_DIR + "seams.png").c_str(), width, height, channels, image, width * channels);
}

/**
 * @brief The entry point of the program.
 * 
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 */
int main(int argc, char* argv[]) {
    check_and_print_error(argc == 3, "Invalid number of arguments. Usage: ./scarving <image_path> <new_width>");
    
    std::string input_image_path = argv[1];
    std::string output_image_path = OUTPUT_DIR + input_image_path.substr(input_image_path.find_last_of("/\\") + 1);
    uint16_t to_remove = std::stoi(argv[2]);

    int width, height, channels;
    uint8_t* image = stbi_load(input_image_path.c_str(), &width, &height, &channels, COLOR_CHANNELS);

    std::cout << "Channels: " << channels << "\n";

    check_and_print_error(image != nullptr, "Failed to load image");

    uint8_t* energy = sobel(image, {width, height, channels});

    stbi_write_png((OUTPUT_DIR + "energy.png").c_str(), width, height, 1, energy, width);

    std::vector<std::vector<size_t>> seams = find_seams(energy, {width, height, channels}, to_remove);

    visualize_seams(image, {width, height, channels}, seams);

    delete[] energy;
    delete[] image;

    return 0;
}