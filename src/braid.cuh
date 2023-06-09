#pragma once
#include <stdint.h>
#include "main.cuh"

#define E 13
#define D (E+1)
const unsigned int N = 1 << E;
const unsigned int INDEX_MASK = N - 1;
const unsigned int MIN_ADVERSARY_OPERATIONS = 8000000; 
const unsigned ABSOLUTE_SPEEDUP_UPPERBOUND = E*(1 << (E - 1))*E + (N/2) - 1;

const unsigned int OPERATIONS_PER_STEP = 124; 
const unsigned int MIN_ADVERSARY_STEPS = (MIN_ADVERSARY_OPERATIONS + (OPERATIONS_PER_STEP - 1)) / OPERATIONS_PER_STEP;

const unsigned int STEPS_LOWERBOUND = MIN_ADVERSARY_STEPS + ABSOLUTE_SPEEDUP_UPPERBOUND; 
const unsigned int SIZE = (STEPS_LOWERBOUND + (E - 1)) / E;

#define FRAGMENT_BYTES 8
//typedef uint8_t fragment_t[FRAGMENT_BYTES];
//typedef struct{
//    uint8_t bytes[FRAGMENT_BYTES];
//} fragment_t;

#define size_fragment_t (FRAGMENT_BYTES * sizeof(uint8_t))

#define GROUP_SIZE 1
const unsigned int BLOCK_BYTE_SIZE =  N*FRAGMENT_BYTES; 
const unsigned int GROUP_BYTE_SIZE = BLOCK_BYTE_SIZE * GROUP_SIZE;

#define INIT_SIZE 4
#define INIT_MASK (INIT_SIZE - 1)

typedef uint8_t init_group_t[INIT_SIZE*FRAGMENT_BYTES];
#define size_init_group_t (INIT_SIZE * size_fragment_t)
#define index_init_group_t(i,j)  (i*FRAGMENT_BYTES + j)

typedef uint8_t block_group_t[N*FRAGMENT_BYTES];
#define size_block_group_t (N * size_fragment_t)
#define index_block_group_t(i,j)  (i*FRAGMENT_BYTES + j)

typedef uint8_t iter_init_group_t[ITER*INIT_SIZE*FRAGMENT_BYTES];
#define M_INIT_SIZE (sizeof(iter_init_group_t))
#define index_iter_init_group_t(i,j,k)  (i*INIT_SIZE*FRAGMENT_BYTES + j*FRAGMENT_BYTES + k)

typedef uint8_t iter_block_group_t[ITER*N*FRAGMENT_BYTES];
#define M_BLOCK_SIZE (sizeof(iter_block_group_t))
#define index_iter_block_group_t(i,j,k)  (i*N*FRAGMENT_BYTES + j*FRAGMENT_BYTES + k)

__global__ void braid(iter_init_group_t, iter_block_group_t);
