#!/bin/bash
make

# Define the problem size
n_particles=10000

# Thread counts to test
threads=(1 2 4 8 16 32 64)

# Loop through each thread count and run the program
for num_threads in "${threads[@]}"; do
    export OMP_NUM_THREADS=$num_threads
    for i in {1..3}; do
        echo "Running with OMP_NUM_THREADS=$num_threads, -n $n_particles (Iteration $i)"
        ./openmp -s 1 -n $n_particles  # Run the program
    done
done
