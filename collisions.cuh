#define DIM 2
#define DIM_2 4
#define DIM_3 9
#define L 8
#define NUM_RADICES 256
#define NUM_BLOCKS 16
#define GROUPS_PER_BLOCK 12
#define THREADS_PER_GROUP 16
#define PADDED_BLOCKS 16
#define PADDED_GROUPS 256

#ifndef UINT32_MAX
  #define UINT32_MAX 0xffffffff
#endif

void cudaInitCells(uint32_t *cells, unsigned int *objects, float *positions,
                   float *dims, unsigned int num_objects, float cell_dim,
                   unsigned int num_blocks, unsigned int threads_per_block);
void cudaInitObjects(float *positions, float *velocities, float *dims,
                     unsigned int num_objects, float max_velocity,
                     float max_dim, unsigned int num_blocks,
                     unsigned int threads_per_block);
void cudaSortCells(uint32_t *cells_in, unsigned int *objects_in,
                   uint32_t *cells_out, unsigned int *objects_out,
                   uint32_t *radices, uint32_t *radix_sums,
                   unsigned int num_objects);
