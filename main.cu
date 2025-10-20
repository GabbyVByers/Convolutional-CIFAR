
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <SFML/Network.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

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
};

int main()
{
    
    CIFAR_DataSet CIFAR;
    
    
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

