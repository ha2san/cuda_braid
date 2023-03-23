#include <iostream>
#include <chrono>

#include "check.cuh"
#include "braid.cuh"

#define RUNS 10 
#define ITER 32

__global__ init_group_array(init_group_t d_x)
{
    //x[j][k] = ((j) * blockgen::FRAGMENT_BYTES + k + 727) as u8;
    int j = blockIdx.x;
    int k = threadIdx.x;
    d_x[j][k] = (uint8_t) ((j) * FRAGMENT_BYTES + k + 727);
}


int main() {
    std::cout << "Invoking CUDA kernel" << std::endl;
    if (!works(true)) {
        std::cout << "CUDA doesn't work" << std::endl;
    }else
    {
        std::cout << "Initializing..." << std::endl;
        init_group_t d_x;
        cudaMalloc((void**)&d_x,size_init_group_t);
        init_group_array<<<INIT_SIZE,FRAGMENT_BYTES>>>(d_x);

        std::cout << "Running..." << std::endl;
        for (int i = 0; i < RUNS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            braid<<<ITER,1>>>(d_x);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Runtime per block: " << elapsed.count() << " seconds for " << ITER << "\"braid\" in parralel" << std::endl;
        }
    }
    return 0;
}
