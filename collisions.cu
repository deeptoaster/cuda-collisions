#include <climits>
#include <cuda_runtime.h>
#include <curand.h>
#include "collisions.cuh"

__global__ void InitCellKernel(unsigned int *cells, unsigned int *objects, 
                               float *positions, float *dims, unsigned int n,
                               float cell_dim) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
      i += gridDim.x * blockDim.x) {
    unsigned int hash = 0;
    unsigned int sides = 0;
    int h;
    int m = 1;
    int q;
    int r;
    float x;
    float a;
    
    // find home cell
    for (int j = 0; j < DIM; j++) {
      x = positions[n * j + i];
      hash = hash << 8 | (unsigned int) (x / cell_dim);
      x -= floor(x / cell_dim) * cell_dim;
      a = dims[n * j + i];
      sides <<= 2;
      
      if (x < a) {
        sides |= 3;
      } else if (cell_dim - x < a) {
        sides |= 1;
      }
    }
    
    cells[i] = hash;
    
    // bit 0 set indicates home cell
    objects[i] = i << 2 | 1;
    
    // find phantom cells
    h = i;
    
    for (int j = 0; j < DIM_3; j++) {
      hash = 0;
      q = hash;
      
      for (int k = 0; k < DIM; k++) {
        r = q % 3 - 1;
        x = positions[n * j + i];
        
        if (r && (sides >> (DIM - k - 1) * 2 & 3 ^ r) & 3) {
          hash = UINT_MAX;
          break;
        }
        
        hash = hash << 8 | (unsigned int) (x / cell_dim) + r;
        q /= 3;
      }
      
      if (hash != UINT_MAX) {
        h += n;
        
        cells[h] = hash;
        
        // bit 0 unset indicates phantom cell
        objects[h] = i << 2;
        
        // keep track of number of cells occupied
        m++;
      }
    }
    
    // fill up remaining cells
    while (m < DIM_2) {
      h += n;
      cells[h] = UINT_MAX;
      objects[h] = i << 2;
      m++;
    }
  }
}

__global__ void ScaleOffsetKernel(float *arr, float scale, float offset,
                                  unsigned int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
      i += gridDim.x * blockDim.x) {
    arr[i] = arr[i] * scale + offset;
  }
}

/**
 * @brief Constructs cell array.
 * @param cells array of cells.
 * @param objects array of corresponding objects.
 * @param positions array of object positions.
 * @param dims array of object sizes
 */
void cudaInitCells(unsigned int *cells, unsigned int *objects,
                   float *positions, float *dims, unsigned int num_objects,
                   float cell_dim, unsigned int num_blocks,
                   unsigned int threads_per_block) {
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
