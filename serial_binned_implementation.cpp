#include "common.h"
#include <cmath>
#include <algorithm>  

// Linked list node for particles in bins
struct particle_node_t {
    particle_t* particle;
    particle_node_t* next;
};

// Bin structure using linked list
struct bin_t {
    particle_node_t* head;
};

// Global variables
int num_bins;           // Number of bins per side
double bin_size;        // Size of each bin
bin_t* bins;           // Array of bins
particle_node_t* nodes; // Pre-allocated nodes for all particles

// Convert particle position to bin index
inline int get_bin_index(double pos, double size) {
    int bin = static_cast<int>(pos / bin_size);
    return std::min(bin, num_bins - 1);
}

// Get 1D bin index from 2D coordinates
inline int get_bin_1d(int x, int y) {
    return y * num_bins + x;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // Initialize spatial binning
    bin_size = cutoff;
    num_bins = static_cast<int>(size / bin_size) + 1;
    
    // Allocate bins and initialize heads to null
    bins = new bin_t[num_bins * num_bins];
    for (int i = 0; i < num_bins * num_bins; i++) {
        bins[i].head = nullptr;
    }
    
    // Pre-allocate nodes for all particles
    nodes = new particle_node_t[num_parts];
}

// Apply the force from neighbor to particle
inline void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Move particle and handle boundary conditions
inline void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Clear all bins (just set heads to null)
    for (int i = 0; i < num_bins * num_bins; i++) {
        bins[i].head = nullptr;
    }

    // Assign particles to bins using pre-allocated nodes
    for (int i = 0; i < num_parts; i++) {
        int bx = get_bin_index(parts[i].x, size);
        int by = get_bin_index(parts[i].y, size);
        int bin_id = get_bin_1d(bx, by);
        
        // Set up node
        nodes[i].particle = &parts[i];
        nodes[i].next = bins[bin_id].head;
        bins[bin_id].head = &nodes[i];
    }

    // Reset accelerations
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = parts[i].ay = 0;
    }

    // Compute forces between particles in neighboring bins
    for (int bx = 0; bx < num_bins; bx++) {
        for (int by = 0; by < num_bins; by++) {
            int bin_id = get_bin_1d(bx, by);
            
            // Loop over all particles in current bin
            for (particle_node_t* curr = bins[bin_id].head; curr != nullptr; curr = curr->next) {
                particle_t& particle = *curr->particle;
                
                // Interact with other particles in same bin
                for (particle_node_t* other = curr->next; other != nullptr; other = other->next) {
                    apply_force(particle, *other->particle);
                    apply_force(*other->particle, particle);
                }
                
                // Interact with particles in neighboring bins
                for (int nx = std::max(0, bx-1); nx <= std::min(num_bins-1, bx+1); nx++) {
                    for (int ny = std::max(0, by-1); ny <= std::min(num_bins-1, by+1); ny++) {
                        if (nx == bx && ny == by) continue; // Skip current bin
                        
                        int neighbor_bin_id = get_bin_1d(nx, ny);
                        for (particle_node_t* neighbor = bins[neighbor_bin_id].head; 
                             neighbor != nullptr; 
                             neighbor = neighbor->next) {
                            apply_force(particle, *neighbor->particle);
                        }
                    }
                }
            }
        }
    }

    // Move particles
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }
}

// Clean up allocated memory
void cleanup_simulation() {
    delete[] bins;
    delete[] nodes;
    bins = nullptr;
    nodes = nullptr;
}
