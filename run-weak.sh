#!/bin/bash
make

# Define the number of threads
threads_list=(1 2 4 8 16 32 64)

# Define the ratios for #particles/#threads (Line 1: 1000, Line 2: 2000, Line 3: 5000, Line 4: 10000)
ratios=(1000 2000 5000 10000)

# Loop through each ratio and the corresponding thread counts
for ratio in "${ratios[@]}"
do
    echo "Running with #particles/#threads = $ratio"
    
    # Loop through the number of threads
    for threads in "${threads_list[@]}"
    do
        # Calculate the number of particles
        particles=$((threads * ratio))

        # Run the simulation once for each (threads, particles)
        echo "Running with OMP_NUM_THREADS=$threads, -n $particles"
        export OMP_NUM_THREADS=$threads
        ./openmp -s 1 -n $particles  # Run the program with the calculated number of particles
    done

    # Add a separator between different ratio runs for clarity
    echo "Finished running for #particles/#threads = $ratio"
done

