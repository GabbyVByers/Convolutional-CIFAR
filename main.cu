
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

struct Image {

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






class ConvolutionalLayer {
public:
    float* input_matrix = nullptr;
    int    input_width  = -1;

    float* kernel_weights = nullptr;
    int    num_kernels  = -1;
    int    kernel_width = -1;

    float* output_matrix = nullptr;
    int    output_width  = -1;

    ConvolutionalLayer(float* input_matrix,
                       int    input_width,
                       int    num_kernels,
                       int    kernel_width) {

        this->input_matrix = input_matrix;
        this->input_width  = input_width;
        this->num_kernels  = num_kernels;
        this->kernel_width = kernel_width;

        assert(kernel_width % 2 == 1);
        assert(input_width  % 2 == 0);
        output_width   = input_width  - kernel_width + 1;
        output_matrix = new float[output_width * output_width];
        kernel_weights = new float[num_kernels * kernel_width * kernel_width];
    }

    ~ConvolutionalLayer() {
        delete[] kernel_weights;
        delete[] output_matrix;
    }

    void execute_forward_pass() {
        forward_pass_kernel<<<1, size>>> ();
    }

    __global__ void forward_pass_kernel(float* input, float* output) {
        size_t inputIndex
    }
};


class ConvolutionalNeuralNetwork {

};



int main()
{
    
    CIFAR_DataSet CIFAR;

    Tensor3 input_activations(3, 32, 32);
    
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

