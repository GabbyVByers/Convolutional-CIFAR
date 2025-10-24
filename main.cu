
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

#define uint unsigned int

__global__ void convolution_forward_pass_kernel(float* input_tensor, float* output_tensor, float* kernel_weights, uint Kw) {
    
    uint x_dim = blockDim.x;
    uint y_dim = blockDim.y;
    uint z_dim = blockDim.z;

    uint x = threadIdx.x;
    uint y = threadIdx.y;
    uint z = threadIdx.z;

    uint N = gridDim.x;
    uint n = blockIdx.x;

    uint x_dim_in = x_dim + Kw - 1;
    uint y_dim_in = y_dim + Kw - 1;

    float* input_matrix  = &input_tensor[z * x_dim_in * y_dim_in];
    float* output_matrix = &output_tensor[(z * N * x_dim * y_dim) + (n * x_dim * y_dim)];
    float* curr_kernel   = &kernel_weights[n * Kw * Kw];

    float sum = 0.0f;
    for (uint i = 0; i < Kw; i++) {
        for (uint j = 0; j < Kw; j++) {
            float activation = input_matrix[((y + j) * x_dim_in) + (x + i)];
            float weight = curr_kernel[(j * Kw) + i];
            sum += activation * weight;
        }
    }

    output_matrix[(y * x_dim) + x] = sum;

}

class ConvolutionalLayer {
public:
    Tensor3* input   = nullptr;
    Tensor3* output  = nullptr;
    Tensor3* kernels = nullptr;
    uint N  = 0;
    uint Kw = 0;

    ConvolutionalLayer(Tensor3* input, Tensor3* output, uint Kw, uint N) {
        assert(output->x_dim == (input->x_dim - Kw + 1));
        assert(output->y_dim == (input->y_dim - Kw + 1));
        assert(output->z_dim == (input->z_dim * N));
        assert((input->x_dim % 2) == 0);
        assert((input->y_dim % 2) == 0);
        assert((Kw % 2) == 1);
        this->input  = input;
        this->output = output;
        this->N  = N;
        this->Kw = Kw;
        kernels = new Tensor3(Kw, Kw, N);
    }

    ~ConvolutionalLayer() {
        delete kernels;
    }

    void execute_forward_pass() {
        float* dev_input  = input->device_data;
        float* dev_output = output->device_data;
        float* dev_kernel = kernels->device_data;
        dim3 BLOCKS_PER_GRID   = dim3(N, 0, 0);
        dim3 THREADS_PER_BLOCK = dim3(input->x_dim - Kw + 1, input->y_dim - Kw + 1, input->z_dim);
        convolution_forward_pass_kernel <<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>> (dev_input, dev_output, dev_kernel, Kw);
    }
};

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

class VisualTensor3 {
    Tensor3* tensor = nullptr;

    VisualTensor3(Tensor3* tensor) {
        this->tensor = tensor;
    }

    void visualize_tensor(sf::RenderWindow& window, uint width) {

    }
};

int main()
{

    Tensor3 tensor_0(32, 32,  3);
    Tensor3 tensor_1(24, 24, 24);
    Tensor3 tensor_2(12, 12, 24);
    
    
    ConvolutionalLayer convolutional_layer_one(&tensor_0, &tensor_1, 9, 8);



    sf::RenderWindow window;
    window.create(sf::VideoMode({ 1920, 1080 }), "Convolutional CIFAR - Gabitha V. Byers", sf::Style::Close);
    window.setFramerateLimit(50);

    sf::Texture cube_texture("Sprites/iso_cube.png");
    sf::Sprite cube_sprite(cube_texture);

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.clear(sf::Color::Black);
        window.display();
    }

    return 0;
}

