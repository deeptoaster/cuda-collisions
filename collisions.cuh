#define DIM 2
#define DIM_2 4
#define DIM_3 9

void cudaInitCells(unsigned int *cells, unsigned int *objects,
                   float *positions, float *dims, unsigned int num_objects,
                   float cell_dim, unsigned int num_blocks,
                   unsigned int threads_per_block);
void cudaInitObjects(float *positions, float *velocities, float *dims,
                     unsigned int num_objects, float max_velocity,
                     float max_dim, unsigned int num_blocks,
                     unsigned int threads_per_block);
