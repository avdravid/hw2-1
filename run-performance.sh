#!/bin/bash
make
#export OMP_NUM_THREADS=64
#./openmp -s 1 -n 100000


# Set OpenMP threads once
export OMP_NUM_THREADS=64

# Define particle counts to iterate over
particles=(1 10 100 1000 10000 100000)

# Run ./openmp for each particle count, repeating 3 times
for n in "${particles[@]}"; do
    for i in {1..3}; do
        echo "Running with OMP_NUM_THREADS=$OMP_NUM_THREADS, -n $n (Iteration $i)"
        ./openmp -s 1 -n "$n"
    done
done

