#include "common.h"
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

//Here we define the Data Structure for storing particles in bins (linked list)
// This was an explicit design choice after using Vectors
struct particle_node_t {
    int particle_id;       //This stores the idex of the particle in the main array
    particle_node_t* next; //Pointer to the next particle in the bin
};

//All of our bins will store a linked list of particles + a lock for safe parallel access
//The lock helps prevent any race conditions
//How? It ensures that when one thread is updating a bin, no other thread can modify it until the first thread is done.
struct bin_t {
    particle_node_t* head; 
    omp_lock_t lock;       //Lock for each thread
};

//Vars for binning and the particles
static int num_bins;               //Num bins along one axis
static double bin_size;             //Size of each bin
static double global_size;          // Total simulation space

static bin_t* bins = nullptr;       //Array of bins
//the pre allocated nodes to avoid dynamic memory allocation each step (makes it faster)
static particle_node_t* nodes = nullptr; // Pre-allocated nodes for all particles

//Stores each particle's bin to avoid recomputing it to reduce computational overhead
static int* particle_bin_ids = nullptr;  //cached bin id for each particle

//Bin index for a given position, this makes sure the particle stays within bounds
inline int get_bin_index(double pos) {
    int b = static_cast<int>(pos / bin_size);
    return (b < 0) ? 0 : (b >= num_bins) ? num_bins - 1 : b;
}

//2d to 1d
inline int get_bin_1d(int bx, int by) {
    return bx * num_bins + by;
}

//method to init the bins, locks, and preallocated structures
void init_simulation(particle_t* parts, int num_parts, double size) {
    global_size = size;
    bin_size = cutoff;
    num_bins = static_cast<int>(size / bin_size) + 1;
    if (num_bins < 1) num_bins = 1;

    //we allocate and initialize bins for the simulation
    bins = new bin_t[num_bins * num_bins];
    for (int i = 0; i < num_bins * num_bins; i++) {
        bins[i].head = nullptr;
        omp_init_lock(&bins[i].lock); //here we nitialize locks for each bin
        //This will help us reduce the race conditions and therefore a segmentation fault error
    }

    //pre allocate linked list nodes 
    nodes = new particle_node_t[num_parts];

    //initialize array to store which bin each particle is in
    particle_bin_ids = new int[num_parts];
}

//We simply compute the force between two particles and updates their acceleration
void apply_force(particle_t &particle, particle_t &neighbor) {

    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;

    double r2 = dx * dx + dy * dy;

    //We have a cutoff for force interaction
    //if particles are too far apart we dont compute it
    if (r2 > cutoff * cutoff) return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;

    //Use atomic operations to avoid race conditions when updating forces
    //Why does this help
    //Atomic operations ensure that a read-modify-write operation happens
    //in a single step this makes it so no other thread can interrupt it while it's executing
    #pragma omp atomic
    particle.ax += coef * dx;
    #pragma omp atomic
    particle.ay += coef * dy;
    
    #pragma omp atomic
    neighbor.ax -= coef * dx;
    #pragma omp atomic
    neighbor.ay -= coef * dy;
}

//Method to update particle's velocity and position for boundary
inline void move(particle_t &p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    //Bounce off the walls if the particle moves out of "bounds"
    while (p.x < 0 || p.x > size) {
        p.x = (p.x < 0) ? -p.x : 2*size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = (p.y < 0) ? -p.y : 2*size - p.y;
        p.vy = -p.vy;
    }
}
//Meat of the program
//Sims one step this inclues binning, force calculations, and particke movement
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    //parallel reset bins for each
    #pragma omp for
    for (int i = 0; i < num_bins * num_bins; i++) {
        bins[i].head = nullptr;
    }

    //parallel assign particles to bins and locked
    #pragma omp for
    for (int i = 0; i < num_parts; i++) {

        int bx = get_bin_index(parts[i].x);
        int by = get_bin_index(parts[i].y);
        int bin_id = get_bin_1d(bx, by);

    
        nodes[i].particle_id = i;

        //We bin and lock
        omp_set_lock(&bins[bin_id].lock);
        nodes[i].next = bins[bin_id].head;
        bins[bin_id].head = &nodes[i];
        omp_unset_lock(&bins[bin_id].lock);

        //we save the bin assignment for each particle
        particle_bin_ids[i] = bin_id;
    }

    //We reset the accelerations for all particles
    #pragma omp for
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;
    }

    //Parallel force dif calculations between particles
    #pragma omp for schedule(static)
    for (int i = 0; i < num_parts; i++) {
        int bin_id = particle_bin_ids[i];
        int bx = bin_id / num_bins;
        int by = bin_id % num_bins;

        //We check bin and its neighbors
        for (int nx = bx - 1; nx <= bx + 1; nx++) {
            if (nx < 0 || nx >= num_bins) continue;

            for (int ny = by - 1; ny <= by + 1; ny++) {
                if (ny < 0 || ny >= num_bins) continue;

                int neighbor_bin_id = get_bin_1d(nx, ny);

                //loop through the linked list of particles in the bin
                for (particle_node_t* neighbor = bins[neighbor_bin_id].head; 
                     neighbor != nullptr; 
                     neighbor = neighbor->next) {
                    
                    int j = neighbor->particle_id;
                    //We ensure it only counts each pair once
                    //no double counting
                    if (i < j) { 
                        //call force method
                        apply_force(parts[i], parts[j]);
                    }
                }
            }
        }
    }

    //We parallel-ly update particles based on the forces
    #pragma omp for
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }
}

//We make sure to clear all allocated memory and to also destroys lock on bins
void cleanup_simulation() {
    if (bins != nullptr) {
        for (int i = 0; i < num_bins * num_bins; i++) {
            omp_destroy_lock(&bins[i].lock);
        }
        delete[] bins;
    }
    delete[] nodes;
    delete[] particle_bin_ids;
    bins = nullptr;
    nodes = nullptr;
    particle_bin_ids = nullptr;
}
