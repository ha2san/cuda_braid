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

__global__ void braid(iter_init_group_t initss,iter_block_group_t blocks)
{
    uint8_t* inits;
    uint8_t* block;
    size_t index_braid = blockIdx.x*blockDim.x + threadIdx.x;
    inits = &initss[index_braid*INIT_SIZE*FRAGMENT_BYTES];
    block = &blocks[index_braid*N*FRAGMENT_BYTES];

    for(size_t i = 0; i<N;i++)
    {
        custom_memcpy( inits+((i & INIT_MASK))*FRAGMENT_BYTES,block+i*FRAGMENT_BYTES, size_fragment_t);
    }
//    if(index_braid == 1)
//        printf("block after first memcpy %d\n",block[0]);
    size_t start = N - (SIZE % N);

    uint8_t buffer[FRAGMENT_BYTES] = {0};
    uint8_t key[16] = {0};

    for(size_t i = 0; i < SIZE; i++)
    {
        size_t index = (i+start) % N;
        for (size_t j = 0; j < D; j++)
        {
            size_t jump = 1 << j;
            size_t target = (index + N - jump) & INDEX_MASK;
            update_key(key,buffer);
            siphash(block+target*FRAGMENT_BYTES,FRAGMENT_BYTES,key,buffer,FRAGMENT_BYTES);
            //if(index_braid == 1 && i == 0 && j == 0)
            //    printf("buffer is %d\n", buffer[0]);
        }

        custom_memcpy( buffer,block+(index)*FRAGMENT_BYTES, size_fragment_t);
    }
//    if(index_braid == 1)
//        printf("block after second memcpy %d\n",block[0]);
}




