#include "common.h"
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// ===============================
// Linked List Structure for Bins
// ===============================

struct particle_node_t {
    int particle_id;              // Store particle index (not a pointer for cache efficiency)
    particle_node_t* next;        // Pointer to next particle in the bin
};

struct bin_t {
    particle_node_t* head;        // Head of linked list
    omp_lock_t lock;              // Lock for thread safety
};

// ===============================
// Global Variables
// ===============================

static int num_bins;
static double bin_size;
static double global_size;

static bin_t* bins = nullptr;          // Array of bins
static particle_node_t* nodes = nullptr; // Pre-allocated nodes for all particles
static int* particle_bin_ids = nullptr; // Cached bin IDs for all particles

// ===============================
// Utility Functions
// ===============================

// Convert position -> bin index
inline int get_bin_index(double pos) {
    int b = static_cast<int>(pos / bin_size);
    return (b < 0) ? 0 : (b >= num_bins) ? num_bins - 1 : b;
}

// Convert 2D bin coordinates -> 1D index
inline int get_bin_1d(int bx, int by) {
    return bx * num_bins + by;
}

// ===============================
// Initialization
// ===============================

void init_simulation(particle_t* parts, int num_parts, double size) {
    global_size = size;
    bin_size = cutoff;
    num_bins = static_cast<int>(size / bin_size) + 1;
    if (num_bins < 1) {
        num_bins = 1; // safety check
    }
    
    // Allocate bins
    bins = new bin_t[num_bins * num_bins];
    for (int i = 0; i < num_bins * num_bins; i++) {
        bins[i].head = nullptr;
        omp_init_lock(&bins[i].lock);
    }

    // Pre-allocate nodes (1 per particle)
    nodes = new particle_node_t[num_parts];

    // Allocate bin ID cache
    particle_bin_ids = new int[num_parts];
}

// ===============================
// Apply Force Function
// ===============================

void apply_force(particle_t &particle, particle_t &neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;

    // Use atomic operations to ensure correctness in parallel execution
    #pragma omp atomic
    particle.ax += coef * dx;
    #pragma omp atomic
    particle.ay += coef * dy;
    
    #pragma omp atomic
    neighbor.ax -= coef * dx;
    #pragma omp atomic
    neighbor.ay -= coef * dy;
}

// ===============================
// Move Function
// ===============================

inline void move(particle_t &p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;

    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2*size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2*size - p.y;
        p.vy = -p.vy;
    }
}

// ===============================
// Simulation Step
// ===============================

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Step 1: Clear all bins (parallelized)
    #pragma omp for
    for (int i = 0; i < num_bins * num_bins; i++) {
        bins[i].head = nullptr; // Reset linked list head
    }

    // Step 2: Assign particles to bins (parallelized)
    #pragma omp for
    for (int i = 0; i < num_parts; i++) {
        int bx = get_bin_index(parts[i].x);
        int by = get_bin_index(parts[i].y);
        int bin_id = get_bin_1d(bx, by);

        // Set up node data
        nodes[i].particle_id = i;
        
        // Use fine-grained locking to add to bin safely
        omp_set_lock(&bins[bin_id].lock);
        nodes[i].next = bins[bin_id].head;
        bins[bin_id].head = &nodes[i];
        omp_unset_lock(&bins[bin_id].lock);

        // Store bin ID for later use
        particle_bin_ids[i] = bin_id;
    }

    // Step 3: Reset particle accelerations (parallelized)
    #pragma omp for
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;
    }

    // Step 4: Compute forces in parallel
    #pragma omp for schedule(static)
    for (int i = 0; i < num_parts; i++) {
        int bin_id = particle_bin_ids[i];
        int bx = bin_id / num_bins;
        int by = bin_id % num_bins;

        for (int nx = bx - 1; nx <= bx + 1; nx++) {
            if (nx < 0 || nx >= num_bins) continue;

            for (int ny = by - 1; ny <= by + 1; ny++) {
                if (ny < 0 || ny >= num_bins) continue;

                int neighbor_bin_id = get_bin_1d(nx, ny);

                // Traverse linked list for neighbor bin
                for (particle_node_t* neighbor = bins[neighbor_bin_id].head; neighbor != nullptr; neighbor = neighbor->next) {
                    int j = neighbor->particle_id;
                    if (i < j) {
                        apply_force(parts[i], parts[j]);
                    }
                }
            }
        }
    }

    // Step 5: Move particles (parallelized)
    #pragma omp for
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }
}

// ===============================
// Cleanup
// ===============================

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
