#include <iostream>
#include <chrono>

#include "check.cuh"
#include "braid.cuh"

#define RUNS 10 
#define ITER 512 

__global__ void init_group_array(init_group_t d_x)
{
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
        init_group_t* d_x,*h_x;
        block_group_t* d_res, *h_res;
        cudaMalloc((void**)&d_x,size_init_group_t*ITER);
        cudaMalloc((void**)&d_res,size_block_group_t*ITER);
	h_x = (init_group_t*) malloc(size_init_group_t*ITER);
	h_res = (block_group_t*)malloc(size_block_group_t*ITER);

	for(size_t i = 0; i<ITER; i++)
	{
        	init_group_array<<<INIT_SIZE,FRAGMENT_BYTES>>>(d_x[i]);
	}

        std::cout << "Running..." << std::endl;
        for (int i = 0; i < RUNS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            braid<<<ITER,1>>>(d_x,d_res);
	    cudaMemcpy(h_x, d_x, size_init_group_t*ITER, cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_res, d_res, size_block_group_t*ITER, cudaMemcpyDeviceToHost);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Runtime per block: " << elapsed.count() << " seconds for " << ITER << " \"braids\" in parralel" << std::endl;
        }
    }
    return 0;
}
