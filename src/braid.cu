#include "braid.cuh"
#include "siphash.cuh"
#include <stdio.h>

__device__ void custom_memcpy(uint8_t* in, uint8_t* out, size_t byte_length)
{
	for(size_t i = 0; i < byte_length; i++)
	{
		out[i] = in[i];
	}
}

__device__ void update_key(uint8_t* key, const uint8_t* buffer)
{
    for(size_t i = 0; i < FRAGMENT_BYTES; i++)
    {
        key[i] = buffer[i];
    }

    for(size_t i = 8; i < 16; i++)
    {
        key[i] = key[i-8];
    }
}

__global__ void braid(struct multiple_init_group* initss,struct multiple_block_group* blocks)
{
    //printf("I'm in braid\n");
    init_group_t* inits;
    block_group_t* block;
    size_t index = blockIdx.x;
    //inits = &initss[blockIdx.x*INIT_SIZE];
    inits = &(initss->array[index]);
    //block = &blocks[blockIdx.x*N];
    block = &(blocks->array[index]);
    for(size_t i = 0; i<N;i++)
    {
	custom_memcpy((uint8_t*)(*block)[i], (uint8_t*)(*inits)[i & INIT_MASK], size_fragment_t);
        //cudaMemcpy(&block[i], &inits[i & INIT_MASK], size_fragment_t, cudaMemcpyDeviceToDevice);
    }
    size_t start = N - (SIZE % N);

    uint8_t buffer[FRAGMENT_BYTES];
    uint8_t key[16];

    for(size_t i = 0; i < SIZE; i++)
    {
        size_t index = (i+start) % N;
        for (size_t j = 0; j < D; j++)
        {
            size_t jump = 1 << j;
            size_t target = (index + N - jump) & INDEX_MASK;
            update_key(key,buffer);
            siphash((void*)(*block)[target],FRAGMENT_BYTES,key,buffer,FRAGMENT_BYTES);
        }

        custom_memcpy((uint8_t*)(*block)[index], (uint8_t*)&buffer, size_fragment_t);
        //cudaMemcpy(&block[index], &buffer, size_fragment_t, cudaMemcpyDeviceToDevice);
    }
}




