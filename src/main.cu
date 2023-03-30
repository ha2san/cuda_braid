#include <iostream>
#include <chrono>

#include "check.cuh"
#include "braid.cuh"
#include "main.cuh"



__global__ void init_group_array(init_group_t d_x)
{
    size_t j = blockIdx.x;
    size_t k = threadIdx.x;
    d_x[j][k] = (uint8_t) ((j) * FRAGMENT_BYTES + k + 727);
}

__global__ void chain_computation(struct multiple_init_group* d_x,struct multiple_block_group* d_res)
{
    for (size_t i = 0; i < FRAGMENT_BYTES; i++)
    {
        (d_x->array)[blockIdx.x][threadIdx.x][i] = d_res->array[blockIdx.x][N-1-threadIdx.x][i];
    }
}

void checks_error(const char* fun)
{
    cudaError_t error;
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        printf("Error: %s %s\n",fun, cudaGetErrorString(error));
        exit(-1);
    }
}



int main() {
    std::cout << "Invoking CUDA kernel" << std::endl;
    if (!works(true)) {
        std::cout << "CUDA doesn't work" << std::endl;
    }else
    {
        std::cout << "Initializing..." << std::endl;
        struct multiple_init_group* d_x, *h_x;
        struct multiple_block_group* d_res, *h_res;
        cudaMalloc((void**)&d_x,M_INIT_SIZE);
        cudaMalloc((void**)&d_res,M_BLOCK_SIZE);
        h_x = (struct multiple_init_group*) malloc(M_INIT_SIZE);
        h_res = (struct multiple_block_group*) malloc(M_BLOCK_SIZE);

        for(size_t i = 0; i<ITER; i++)
        {
            init_group_array<<<INIT_SIZE,FRAGMENT_BYTES>>>(d_x->array[i]);
            checks_error("init_group_array");
        }

        chain_computation<<<ITER,INIT_SIZE>>>(d_x,d_res);
        checks_error("chain_computation");
        std::cout << "Running with iter = " << ITER << std::endl;
        for (int i = 0; i < RUNS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            //braid<<<ITER,1>>>(d_x,d_res);
            braid<<<NB_BLOCKS, NB_THREADS>>>(d_x,d_res);
            checks_error("braid");

            chain_computation<<<ITER,INIT_SIZE>>>(d_x,d_res);
            checks_error("chain_computation");
            cudaMemcpy(h_x, d_x, M_BLOCK_SIZE, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_res, d_res, M_INIT_SIZE, cudaMemcpyDeviceToHost);
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Runtime per block: " << elapsed.count() << " ms for " << ITER << " \"braids\" in parralel" << std::endl;
            double elapsed_double = std::chrono::duration<double, std::milli>(elapsed).count();
            double read_speed = (ITER*(GROUP_BYTE_SIZE * 1000.0))/ (elapsed_double * (double)(1<<20));
            std::cout << "Read speed = " << read_speed << "MB/s" << std::endl;
        }
    }
    return 0;
}
