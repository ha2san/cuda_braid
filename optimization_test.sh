#!/bin/bash

# Define variables for the ranges we'll loop over
MIN_BLOCK_EXPONENT=5
MAX_BLOCK_EXPONENT=10
MIN_THREAD_EXPONENT=5
MAX_THREAD_EXPONENT=10

header_file="src/main.cuh"

# Loop over the ranges of block and thread counts
for block_exp in $(seq $MIN_BLOCK_EXPONENT $MAX_BLOCK_EXPONENT); do
  for thread_exp in $(seq $MIN_THREAD_EXPONENT $MAX_THREAD_EXPONENT); do
    # Calculate the actual block and thread counts based on the exponents
    nb_block=$((2**block_exp))
    nb_thread=$((2**thread_exp))

    # Modify the header file with the new values
    sed -i "s/#define NB_BLOCK.*/#define NB_BLOCK $nb_block/" $header_file 
    sed -i "s/#define NB_THREADS.*/#define NB_THREADS $nb_thread/" $header_file 

    # Compile the program
    cmake --build build/ 

    # Run the program and store the output
    ./build/test >> output.txt
  done
done

