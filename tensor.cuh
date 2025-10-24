
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define uint unsigned int

class Tensor3 {
public:
	float* host_data   = nullptr;
	float* device_data = nullptr;

	uint capacity = 0;
	uint x_dim = 0;
	uint y_dim = 0;
	uint z_dim = 0;

	Tensor3(uint x, uint y, uint z) {
		capacity = x * y * z;
		x_dim = x;
		y_dim = y;
		z_dim = z;
		host_data = new float[capacity];
		cudaMalloc((void**)&device_data, capacity * sizeof(float));
	}

	~Tensor3() {
		delete[] host_data;
		cudaFree(device_data);
	}

	void transfer_to_gpu() const {
		cudaMemcpy(device_data, host_data, capacity * sizeof(float), cudaMemcpyHostToDevice);
	}

	void transfer_to_cpu() const {
		cudaMemcpy(host_data, device_data, capacity * sizeof(float), cudaMemcpyDeviceToHost);
	}
};

