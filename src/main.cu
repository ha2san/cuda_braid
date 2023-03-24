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

__global__ void chain_computation(init_group_t* d_x,block_group_t* d_res)
{
    //size_t d_x_index = blockIdx.x * INIT_SIZE + threadIdx.x;
    //size_t d_res_index = blockIdx.x*INIT_SIZE+N-1-threadIdx.x;
    for (size_t i = 0; i < FRAGMENT_BYTES; i++)
    {
        d_x[blockIdx.x*INIT_SIZE][threadIdx.x][i] = d_res[blockIdx.x*INIT_SIZE][N-1-threadIdx.x][i];
    }
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

        chain_computation<<<ITER,INIT_SIZE>>>(d_x,d_res);
        std::cout << "Running..." << std::endl;
        for (int i = 0; i < RUNS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            braid<<<ITER,1>>>(d_x,d_res);
            chain_computation<<<ITER,INIT_SIZE>>>(d_x,d_res);
            cudaMemcpy(h_x, d_x, size_init_group_t*ITER, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_res, d_res, size_block_group_t*ITER, cudaMemcpyDeviceToHost);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = (end - start)/1000.0;
            std::cout << "Runtime per block: " << elapsed.count() << " ms for " << ITER << " \"braids\" in parralel" << std::endl;
            double read_speed = (ITER*(GROUP_BYTE_SIZE * 1000.0))/ (elapsed.count() * (1<<20));
            std::cout << "Read speed = " << read_speed << "MB/s" << std::endl;
        }
    }
    return 0;
}
