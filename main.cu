
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <SFML/Network.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

#include "tensor.cuh"

__global__ void forward_pass_convolution_kernel() {




}

class CIFAR_DataSet {
public:
    CIFAR_DataSet() {
        std::vector<std::ifstream> CIFAR_binary_files;
        CIFAR_binary_files.push_back(std::ifstream("CIFAR/data_batch_1.bin", std::ios::binary));
        CIFAR_binary_files.push_back(std::ifstream("CIFAR/data_batch_2.bin", std::ios::binary));
        CIFAR_binary_files.push_back(std::ifstream("CIFAR/data_batch_3.bin", std::ios::binary));
        CIFAR_binary_files.push_back(std::ifstream("CIFAR/data_batch_4.bin", std::ios::binary));
        CIFAR_binary_files.push_back(std::ifstream("CIFAR/data_batch_5.bin", std::ios::binary));

        for (int i = 0; i < 5; i++) {
            std::ifstream& file = CIFAR_binary_files[i];
            assert(file.is_open());
        }
    }

private:
    unsigned char* host_dataset = nullptr;
    unsigned char* device_dataset = nullptr;
};

class ConvolutionalLayer {
public:
    Tensor3* input          = nullptr;
    Tensor3* output         = nullptr;
    Tensor3* kernel_weights = nullptr;

    int num_kernels  = -1;
    int kernel_width = -1;

    ConvolutionalLayer(Tensor3* input, Tensor3* output, int num_kernels, int kernel_width) {
        this->input        = input;
        this->output       = output;
        this->num_kernels  = num_kernels;
        this->kernel_width = kernel_width;
        assert(output->x_dim == (input->x_dim - kernel_width + 1));
        assert(output->y_dim == (input->y_dim - kernel_width + 1));
        assert(output->z_dim == (input->x_dim * num_kernels));
        kernel_weights = new Tensor3(num_kernels, kernel_width, kernel_width);
    }

    ~ConvolutionalLayer() {
        delete kernel_weights;
    }

    void execute_forward_pass() {
        size_t x_dim = output->x_dim;
        size_t y_dim = output->y_dim;
        size_t z_dim = output->z_dim;
        dim3 BLOCKS_PER_GRID = 1;
        dim3 THREADS_PER_BLOCK = dim3(x_dim, y_dim, z_dim);
        forward_pass_convolution_kernel <<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>> ();
    }
};

int main()
{
    sf::RenderWindow window;
    window.create(sf::VideoMode({ 800, 800 }), "My Window");

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.clear(sf::Color::Blue);
        window.display();
    }

    return 0;
}

