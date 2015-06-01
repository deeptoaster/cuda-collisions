#include <climits>
#include <stdint.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "collisions.cuh"

__global__ void InitCellKernel(uint32_t *cells, unsigned int *objects, 
                               float *positions, float *dims, unsigned int n,
                               float cell_dim) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x *
      blockDim.x) {
    uint32_t hash = 0;
    unsigned int sides = 0;
    int h = i * DIM_2;
    int m = 1;
    int q;
    int r;
    float x;
    float a;
    
    // find home cell
    for (int j = 0; j < DIM; j++) {
      x = positions[n * j + i];
      
      // cell ID is simply the bits of each cell coordinate concatenated
      hash = hash << 8 | (uint32_t) (x / cell_dim);
      
      // determine if the cell is close enogh to overlap cells on the side
      x -= floor(x / cell_dim) * cell_dim;
      a = dims[n * j + i];
      sides <<= 2;
      
      // keep track of which side of the center, if any, the object overlaps
      if (x < a) {
        sides |= 3;
      } else if (cell_dim - x < a) {
        sides |= 1;
      }
    }
    
    cells[h] = hash;
    
    // bit 0 set indicates home cell
    objects[h] = i << 2 | 1;
    
    // find phantom cells in the Moore neighborhood
    for (int j = 0; j < DIM_3; j++) {
      hash = 0;
      
      // run through the components of each potential side cell
      q = j;
      
      for (int k = 0; k < DIM; k++) {
        r = q % 3 - 1;
        x = positions[n * j + i];
        
        // skip this cell if the object is on the wrong side
        if (r && (sides >> (DIM - k - 1) * 2 & 3 ^ r) & 3) {
          hash = UINT32_MAX;
          break;
        }
        
        // cell ID of the neighboring cell
        hash = hash << 8 | (uint32_t) (x / cell_dim) + r;
        q /= 3;
      }
      
      // only add this cell to the list if there's potential overlap
      if (hash != UINT32_MAX) {
        h++;
        
        cells[h] = hash;
        
        // bit 0 unset indicates phantom cell
        objects[h] = i << 2;
        
        // keep track of number of cells occupied
        m++;
      }
    }
    
    // fill up remaining cells
    while (m < DIM_2) {
      h++;
      cells[h] = UINT32_MAX;
      objects[h] = i << 2;
      m++;
    }
  }
}

__global__ void RadixTabulateKernel(uint32_t *keys,
                                    uint32_t *radices,
                                    unsigned int n, int shift) {
  extern __shared__ uint32_t s[];
  unsigned int cells_per_group = n / gridDim.x / GROUPS_PER_BLOCK;
  int group = threadIdx.x / THREADS_PER_GROUP;
  int group_start = (blockIdx.x * GROUPS_PER_BLOCK + group) * cells_per_group;
  int group_end = group_start + cells_per_group;
  uint32_t k;
  
  // initialize shared memory
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < GROUPS_PER_BLOCK *
      NUM_RADICES; i += gridDim.x * blockDim.x) {
    s[i] = 0;
  }
  
  // count instances of each radix
  for (int i = group_start + threadIdx.x % THREADS_PER_GROUP; i < group_end;
      i += THREADS_PER_GROUP) {
    // need only avoid bank conflicts by group
    k = (keys[i] >> shift & NUM_RADICES - 1) * GROUPS_PER_BLOCK + group;
    
    // increment radix counters sequentially by thread in the thread group
    for (int j = 0; j < THREADS_PER_GROUP; j++) {
      if (threadIdx.x % THREADS_PER_GROUP == j) {
        s[k]++;
      }
    }
  }
  
  // copy to global memory
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < GROUPS_PER_BLOCK *
      NUM_RADICES; i += gridDim.x * blockDim.x) {
    radices[(i / GROUPS_PER_BLOCK * gridDim.x + blockIdx.x) *
        GROUPS_PER_BLOCK + i % GROUPS_PER_BLOCK] = s[i];
  }
}

__global__ void ScaleOffsetKernel(float *arr, float scale, float offset,
                                  unsigned int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x *
      blockDim.x) {
    arr[i] = arr[i] * scale + offset;
  }
}

/**
 * @brief Constructs cell array.
 * @param cells array of cells.
 * @param objects array of corresponding objects.
 * @param positions array of object positions.
 * @param dims array of object sizes
 * @param num_objects the number of objects to process.
 * @param cell_dim the size of each cell in any dimension.
 */
void cudaInitCells(uint32_t *cells, unsigned int *objects, float *positions,
                   float *dims, unsigned int num_objects, float cell_dim,
                   unsigned int num_blocks, unsigned int threads_per_block) {
  InitCellKernel<<<num_blocks, threads_per_block>>>(
      cells, objects, positions, dims, num_objects, cell_dim);
}

/**
 * @brief Randomly generates object properties.
 * @param positions array of positions.
 * @param velocities array of velocities.
 * @param dims array of dimensions.
 * @param num_objects the number of objects to generate.
 * @param max_velocity the maximum possible speed in any dimension.
 * @param max_dim the maximum size in any dimension.
 */
void cudaInitObjects(float *positions, float *velocities, float *dims,
                     unsigned int num_objects, float max_velocity,
                     float max_dim, unsigned int num_blocks,
                     unsigned int threads_per_block) {
  curandGenerator_t generator;
  
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  
  // randomly generate positions ranging from (0, 0) to (1, 1)
  curandGenerateUniform(generator, positions, num_objects * DIM);
  
  // randomly generate speeds ranging from -max_velocity to max_velocity
  curandGenerateUniform(generator, velocities, num_objects * DIM);
  ScaleOffsetKernel<<<num_blocks, threads_per_block>>>(
      velocities, max_velocity * 2, -max_velocity, num_objects);
  
  // randomly generate sizes ranging from 0 to max_dim
  curandGenerateUniform(generator, dims, num_objects * DIM);
  ScaleOffsetKernel<<<num_blocks, threads_per_block>>>(
      dims, max_dim, 0, num_objects);
}

/**
 * @brief Radix sorts an array of objects using occupations as keys.
 * @param cells_in the input array of cells.
 * @param objects_in the input array of objects.
 * @param cells_out sorted array of cells.
 * @param objects_out array of objects sorted by corresponding cells.
 * @param radices working array to hold radix data.
 * @param num_objects the number of objects included.
 */
void cudaSortCells(uint32_t *cells_in, unsigned int *objects_in,
                   uint32_t *cells_out, unsigned int *objects_out,
                   uint32_t *radices, unsigned int num_objects) {
  for (int i = 0; i < sizeof(uint32_t) * 8; i += L) {
    RadixTabulateKernel<<<NUM_BLOCKS, GROUPS_PER_BLOCK * THREADS_PER_GROUP,
                          GROUPS_PER_BLOCK * NUM_RADICES * sizeof(uint32_t)>>>(
        cells_in, radices, num_objects * DIM_2, i);
  }
}
