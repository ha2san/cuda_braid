#include <iostream>
#include <chrono>
#include <assert.h>

#include "braid.h"
#include "main.h"



void init_group_array(init_group_t d_x)
{
    for(size_t j = 0; j < INIT_SIZE; j++)
    {
        for(size_t k = 0; k < FRAGMENT_BYTES; k++)
        {
            d_x[j*FRAGMENT_BYTES+k] = (uint8_t) ((j) * FRAGMENT_BYTES + k + 727);
        }
    }
}

void chain_computation(init_group_t d_x,block_group_t d_res)
{
    for (size_t init_i = 0; init_i < INIT_SIZE; init_i++)
    {
        for (size_t i = 0; i < FRAGMENT_BYTES; i++)
        {
            assert(N - 1 - init_i < N );
            size_t index_init = init_i*FRAGMENT_BYTES+i;
            size_t index_block = (N -1 - init_i)*FRAGMENT_BYTES+i;
            (d_x)[index_init] = (d_res)[index_block];
        }
    }
    printf("\n");
}

void show_init(init_group_t x)
{
    printf("show init\n");
    for (size_t init_i = 0; init_i < INIT_SIZE; init_i++)
    {
        for (size_t i = 0; i < FRAGMENT_BYTES; i++)
        {
            std::cout << (int)(x)[init_i*FRAGMENT_BYTES+i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void show_block(block_group_t x)
{
    printf("show first 4 line of block\n");
    for (size_t init_i = 0; init_i < 4; init_i++)
    {
        for (size_t i = 0; i < FRAGMENT_BYTES; i++)
        {
            std::cout << (int)(x)[init_i*FRAGMENT_BYTES+i] << " ";
        }
        std::cout << std::endl;
    }
    printf("show last 4 line of block\n");
    for (size_t init_i = N-4; init_i < N; init_i++)
    {
        for (size_t i = 0; i < FRAGMENT_BYTES; i++)
        {
            std::cout << (int)(x)[init_i*FRAGMENT_BYTES+i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    init_group_t x;
    block_group_t res = {0};
    init_group_array(x);
    show_init(x);
    show_block(res);
    braid(x,res);
    show_block(res);
    show_init(x);
    chain_computation(x,res);
    show_init(x);
    show_block(res);
    double read_speeds[RUNS];
    printf("Running on CPU\n");
    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        show_init(x);
        braid(x,res);
        chain_computation(x,res);
        show_init(x);
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
    show_init(x);
    return 0;
}
