
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

__global__ void convolution_forward_pass_kernel() {

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
        dim3 BLOCKS_PER_GRID = dim3(num_kernels);
        dim3 THREADS_PER_BLOCK = dim3(output->z_dim, output->y_dim, output->x_dim);
        convolution_forward_pass_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>> ();
    }
};

class TensorRenderer {

public:
    sf::RenderWindow* window  = nullptr;
    sf::Texture* cube_texture = nullptr;
    sf::Sprite*  cube_sprite  = nullptr;

    TensorRenderer(sf::RenderWindow* window) {
        this->window = window;
        cube_texture = new sf::Texture("Sprites/iso_cube.png");
        cube_sprite  = new sf::Sprite(*cube_texture);
    }

    void draw(int z_dim, int y_dim, int x_dim, int curr_layer) {
        
        
        
    }
};

int main()
{
    sf::RenderWindow window;
    window.create(sf::VideoMode({ 1920, 1080 }), "Convolutional CIFAR - Gabitha V. Byers", sf::Style::Close);
    window.setFramerateLimit(50);

    TensorRenderer tensor_renderer(&window);

    sf::Texture cube_texture("Sprites/iso_cube.png");
    sf::Sprite cube_sprite(cube_texture);

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.clear(sf::Color::Black);
        //tensor_renderer.draw(10, 10, 10, 1);
        
        
        int z_dim = 3;
        int y_dim = 32;
        int x_dim = 32;
        int curr_layer = 1;

        float window_height = (float)window.getSize().y;
        sf::Vector2f base_position = sf::Vector2f(100, 300);

        for (int x = 0; x < x_dim; x++) {
            for (int y = 0; y < y_dim; y++) {
                for (int z = 0; z < z_dim; z++) {
                    cube_sprite.setPosition(base_position);
                    cube_sprite.move(sf::Vector2f(0, 14 * (y_dim - y)));
                    cube_sprite.move(sf::Vector2f(z * 15, z * 7));
                    cube_sprite.move(sf::Vector2f((x_dim - x) * 15, (x_dim - x) * -7));
                    window.draw(cube_sprite);
                }
            }
        }

        window.display();
    }

    return 0;
}

