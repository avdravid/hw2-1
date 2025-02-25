#include "common.h"
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static int num_bins;             
static double bin_size;         
static double global_size;       

struct bin_t {
    std::vector<int> particles;
};
static bin_t* bins = nullptr;   


static int* particle_bin_ids = nullptr;

inline int get_bin_index(double pos)
{
    // Convert coordinate -> bin index along one dimension
    int b = static_cast<int>(pos / bin_size);
    if (b < 0) return 0;
    if (b >= num_bins) return num_bins - 1;
    return b;
}

inline int get_bin_1d(int bx, int by)
{
    return bx * num_bins + by;
}

void init_simulation(particle_t* parts, int num_parts, double size)
{
    // We'll store the domain size in a global so we can reference it
    global_size = size;

    // We set bin_size to the cutoff from common.h
    bin_size = cutoff;

    // Compute how many bins we need in each dimension
    num_bins = static_cast<int>(size / bin_size) + 1;
    if (num_bins < 1) {
        num_bins = 1; // safety net if size < cutoff
    }

    // Allocate our global array of bins:
    bins = new bin_t[num_bins * num_bins];
    // We can optionally reserve space in each bin's vector
    // but for large n, you'd typically do it more carefully.
    // For a rough heuristic:
    int total_bins = num_bins * num_bins;
    int avg_particles_per_bin = num_parts / (total_bins) + 1;

    for (int i = 0; i < total_bins; i++) {
        bins[i].particles.reserve(avg_particles_per_bin);
    }

    // Allocate our global array for caching each particle's bin id
    particle_bin_ids = new int[num_parts];

    // Nothing else to do here. We'll just reuse these data structures
    // inside simulate_one_step() for each iteration.
}

void apply_force(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff*cutoff) return;

    r2 = fmax(r2, min_r*min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;

    // Use atomic updates to avoid race conditions
    #pragma omp atomic
    particle.ax += coef * dx;
    #pragma omp atomic
    particle.ay += coef * dy;
    
    #pragma omp atomic
    neighbor.ax -= coef * dx;
    #pragma omp atomic
    neighbor.ay -= coef * dy;
}


inline void move(particle_t &p, double size)
{
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
void simulate_one_step(particle_t* parts, int num_parts, double size)
{
    // Step 1: Clear all bins (parallelized)
    #pragma omp for
    for (int i = 0; i < num_bins * num_bins; i++) {
        bins[i].particles.clear();
    }

    // Step 2: Assign particles to bins (parallelized, critical section for push_back)
    #pragma omp for
    for (int i = 0; i < num_parts; i++) {
        int bx = get_bin_index(parts[i].x);
        int by = get_bin_index(parts[i].y);
        int bin_id = get_bin_1d(bx, by);

        // Critical section prevents race conditions when modifying bin contents
        #pragma omp critical
        {
            bins[bin_id].particles.push_back(i);
        }

        // Store bin ID for later use
        particle_bin_ids[i] = bin_id;
    }

    // Step 3: Reset accelerations for all particles (parallelized)
    #pragma omp for
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0.0;
        parts[i].ay = 0.0;
    }

    // Step 4: Compute forces in parallel
    #pragma omp for schedule(static)
    for (int i = 0; i < num_parts; i++) {
        int bin_id = particle_bin_ids[i];
        int bx = bin_id / num_bins;  // Convert 1D index back to 2D
        int by = bin_id % num_bins;

        // Loop over this bin and all 8 surrounding bins
        for (int nx = bx - 1; nx <= bx + 1; nx++) {
            if (nx < 0 || nx >= num_bins) continue;

            for (int ny = by - 1; ny <= by + 1; ny++) {
                if (ny < 0 || ny >= num_bins) continue;

                int neighbor_bin_id = get_bin_1d(nx, ny);
                std::vector<int>& neighbors = bins[neighbor_bin_id].particles;

                for (int j : neighbors) {
                    if (i < j) { // Ensures each particle pair is computed only once
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



void cleanup_simulation()
{
    delete[] bins; 
    bins = nullptr;
    delete[] particle_bin_ids;
    particle_bin_ids = nullptr;
}
