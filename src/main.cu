#include <iostream>
#include <chrono>
#include <assert.h>

#include "check.cuh"
#include "braid.cuh"
#include "main.cuh"



__global__ void init_group_array(uint8_t* d_x, size_t iter)
{
    size_t j = blockIdx.x; // => 0 to INIT_SIZE
    size_t k = threadIdx.x;// => 0 to FRAGMENT_BYTES
    assert(j < INIT_SIZE && "j out of bounds for init_group_array");
    assert(k < FRAGMENT_BYTES && "k out of bounds for init_group_array");
    d_x[j*FRAGMENT_BYTES + k] = (uint8_t) ((j) * FRAGMENT_BYTES + k + 727*iter);
}

__global__ void chain_computation(iter_init_group_t d_x,iter_block_group_t d_res)
{
    size_t iter = blockIdx.x* blockDim.x + threadIdx.x; 
    for (size_t init_i = 0; init_i < INIT_SIZE; init_i++)
    {
        for (size_t i = 0; i < FRAGMENT_BYTES; i++)
        {
            size_t index_init = index_iter_init_group_t(iter,init_i,i);
            size_t index_block = //index_iter_block_group_t(iter,N-1-init_i,i);
                        (N - 1 -init_i)*FRAGMENT_BYTES + i;
            //d_x[index_init] = d_res[index_block]; 
            d_x[index_init] = d_res[index_block+iter*N*FRAGMENT_BYTES]; 
        }
    }
//    if (iter == 1){
//        printf("\n");
//    }
    
}

__global__ void show_init(iter_init_group_t d_x)
{
    size_t iter = blockIdx.x* blockDim.x + threadIdx.x; 
    if(iter == 1)
    {
        printf("show init\n");
        for(size_t init_i = 0; init_i < INIT_SIZE; init_i++)
        {
            for (size_t i = 0; i < FRAGMENT_BYTES; i++)
            {
                printf("%d ",d_x[index_iter_init_group_t(iter,init_i,i)]); 
            }
            printf("\n");
        }
        printf("\n");
    }
}
__global__ void show_block(iter_block_group_t b)
{
    size_t iter = blockIdx.x* blockDim.x + threadIdx.x; 
    if(iter == 1)
    {
        printf("show first 4 line of block\n");
        for(size_t i = 0; i < 4; i++)
        {
            for (size_t j = 0; j < FRAGMENT_BYTES; j++)
            {
                printf("%d ",b[index_iter_block_group_t(iter,i,j)]); 
            }
            printf("\n");
        }
        printf("show last 4 line of block\n");
        for(size_t i = N-4; i < N; i++)
        {
            for (size_t j = 0; j < FRAGMENT_BYTES; j++)
            {
                printf("%d ",b[index_iter_block_group_t(iter,i,j)]); 
            }
            printf("\n");
        }
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
    if (!works(true)) {
        std::cout << "CUDA doesn't work" << std::endl;
    }else
    {
        //iter_init_group_t h_x; 
        //iter_block_group_t h_res;
        uint8_t* h_x = (uint8_t*) malloc(M_INIT_SIZE);
        uint8_t* h_res = (uint8_t*) malloc(M_BLOCK_SIZE); 
        uint8_t* d_x;
        uint8_t* d_res;
        cudaMalloc((void**)&d_x,M_INIT_SIZE);
        cudaMalloc((void**)&d_res,M_BLOCK_SIZE);

        for(size_t i = 0; i<ITER; i++)
        {
            init_group_array<<<INIT_SIZE,FRAGMENT_BYTES>>>(d_x+(i*INIT_SIZE*FRAGMENT_BYTES),i);
            checks_error("init_group_array: ");
        }

        //show_init<<<NB_BLOCKS,NB_THREADS>>>(d_x);

        //show_block<<<NB_BLOCKS,NB_THREADS>>>(d_res);
        braid<<<NB_BLOCKS, NB_THREADS>>>(d_x,d_res);
        checks_error("braid");
        //show_block<<<NB_BLOCKS,NB_THREADS>>>(d_res);
        //show_init<<<NB_BLOCKS,NB_THREADS>>>(d_x);

        chain_computation<<<NB_BLOCKS,NB_THREADS>>>(d_x,d_res);
        checks_error("chain_computation: ");
        //show_init<<<NB_BLOCKS,NB_THREADS>>>(d_x);
        //show_block<<<NB_BLOCKS,NB_THREADS>>>(d_res);
        std::cout << "Running with " << NB_BLOCKS << " blocks and " << NB_THREADS << " threads" << std::endl;
        double read_speeds[RUNS];
        for (int i = 0; i < RUNS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            //show_init<<<NB_BLOCKS,NB_THREADS>>>(d_x);
            braid<<<NB_BLOCKS, NB_THREADS>>>(d_x,d_res);
            checks_error("braid");

            chain_computation<<<NB_BLOCKS,NB_THREADS>>>(d_x,d_res);
            checks_error("chain_computation: ");
            //show_init<<<NB_BLOCKS,NB_THREADS>>>(d_x);
            cudaMemcpy(h_x, d_x, M_BLOCK_SIZE, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_res, d_res, M_INIT_SIZE, cudaMemcpyDeviceToHost);
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            double elapsed_double = std::chrono::duration<double, std::milli>(elapsed).count();
            double read_speed = (ITER*(GROUP_BYTE_SIZE * 1000.0))/ (elapsed_double * (double)(1<<20));
            read_speeds[i] = read_speed;
        }
        double sum = 0;
        for (int i = 0; i < RUNS; i++) {
            sum += read_speeds[i];
        }
        std::cout << "Average read speed = " << sum / RUNS << "MB/s" << std::endl;
        //show_init<<<NB_BLOCKS,NB_THREADS>>>(d_x);
    }
    return 0;
}
