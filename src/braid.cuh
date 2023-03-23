#pragma once
#include <stdint.h>

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
typedef uint8_t fragment_t[FRAGMENT_BYTES];
//typedef struct{
//    uint8_t bytes[FRAGMENT_BYTES];
//} fragment_t;

#define size_fragment_t (FRAGMENT_BYTES * sizeof(uint8_t))

//#define GROUP_SIZE 1
//const unsigned int BLOCK_BYTE_SIZE =  N*8 
//const unsigned int GROUP_BYTE_SIZE = BLOCK_BYTE_SIZE * GROUP_SIZE;

#define INIT_SIZE 4
#define INIT_MASK (INIT_SIZE - 1)
typedef fragment_t init_group_t[INIT_SIZE];
//typedef struct{
//    fragment_t fragments[INIT_SIZE];
//} init_group_t;

#define size_init_group_t (INIT_SIZE * size_fragment_t)

typedef fragment_t block_group_t[N];
//typedef struct{
//    fragment_t fragments[N];
//} block_group_t;
#define size_block_group_t (N * size_fragment_t)

__global__ void braid(init_group_t*, block_group_t*);
