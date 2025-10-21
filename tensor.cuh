
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Tensor3 {
public:
	Tensor3(unsigned int x_dim, unsigned int y_dim, unsigned int z_dim) {
		this->x_dim = x_dim;
		this->y_dim = y_dim;
		this->z_dim = z_dim;
		capacity = x_dim * y_dim * z_dim;
		host_data = new float[capacity];
		cudaMalloc((void**)&device_data, capacity * sizeof(float));
	}

	~Tensor3() {
		delete[] host_data;
		cudaFree(device_data);
	}

	void set_value_host(unsigned int x, unsigned int y, unsigned int z, float value) {
		unsigned int idx = (x_step_size() * x) + (y_step_size() * y) + z;
		host_data[idx] = value;
	}

	float* host_matrix_ptr(unsigned int x) {
		unsigned int idx = x_step_size() * x;
		return &host_data[idx];
	}

	float* device_matrix_ptr(unsigned int x) {
		unsigned int idx = x_step_size() * x;
		return &device_data[idx];
	}

	void transfer_to_gpu() { cudaMemcpy(device_data, host_data, capacity * sizeof(float), cudaMemcpyHostToDevice); }
	void transfer_to_cpu() { cudaMemcpy(host_data, device_data, capacity * sizeof(float), cudaMemcpyDeviceToHost); }

	float* host_data   = nullptr;
	float* device_data = nullptr;
	unsigned int x_dim, y_dim, z_dim;
	unsigned int capacity;

	unsigned int x_step_size() { return z_dim * y_dim; }
	unsigned int y_step_size() { return z_dim; }
};

