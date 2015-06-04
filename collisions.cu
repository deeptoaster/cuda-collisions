#include <climits>
#include <cstdio>
#include <stdint.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "collisions.cuh"

__device__ void dPrefixSum(uint32_t *values, unsigned int n) {
  int offset = 1;
  int a;
  uint32_t temp;
  
  // upsweep
  for (int d = n / 2; d; d /= 2) {
    __syncthreads();
    
    if (threadIdx.x < d) {
      a = (threadIdx.x * 2 + 1) * offset - 1;
      values[a + offset] += values[a];
    }
    
    offset *= 2;
  }
  
  if (!threadIdx.x) {
    values[n - 1] = 0;
  }
  
  // downsweep
  for (int d = 1; d < n; d *= 2) {
    __syncthreads();
    offset /= 2;
    
    if (threadIdx.x < d) {
      a = (threadIdx.x * 2 + 1) * offset - 1;
      temp = values[a];
      values[a] = values[a + offset];
      values[a + offset] += temp;
    }
  }
}

__device__ void dSumReduce(unsigned int *values, unsigned int *out) {
  // wait for the whole array to be populated
  __syncthreads();
  
  // sum by reduction, using half the threads in each subsequent iteration
  unsigned int threads = blockDim.x;
  unsigned int half = threads / 2;
  
  while (half) {
    if (threadIdx.x < half) {
      // only keep going if the thread is in the first half threads
      for (int k = threadIdx.x + half; k < threads; k += half) {
        values[threadIdx.x] += values[k];
      }
      
      threads = half;
    }
    
    half /= 2;
    
    // make sure all the threads are on the same iteration
    __syncthreads();
  }
  
  // only let one thread update the current sum
  if (!threadIdx.x) {
    atomicAdd(out, values[0]);
  }
}

__global__ void cellCollideKernel(uint32_t *cells, uint32_t *objects,
                                  float *positions, float *velocities,
                                  float *dims, unsigned int n, unsigned int m,
                                  unsigned int cells_per_thread,
                                  unsigned int *collision_count) {
  extern __shared__ unsigned int t[];
  
  int thread_start = (blockIdx.x * blockDim.x + threadIdx.x) *
      cells_per_thread;
  int thread_end = thread_start + cells_per_thread;
  int start = -1;
  int i = thread_start;
  uint32_t last = UINT32_MAX;
  uint32_t home;
  uint32_t phantom;
  unsigned int h;
  unsigned int p;
  unsigned int collisions = 0;
  float dh;
  float dp;
  float dx;
  float d;
  
  while (1) {
    if (!blockIdx.x && !threadIdx.x) {
      printf("%d / %d: %d (%d, %d)\n", i, m, cells[i], h, p);
    }
    
    // find cell ID change indices
    if (i >= m || cells[i] >> 1 != last) {
      // at least one home-cell object and at least one other object present
      if (start + 1 && h >= 1 && h + p >= 2) {
        for (int j = start; j < start + h; j++) {
          home = objects[j] >> 1;
          dh = dims[home];
          
          for (int k = j + 1; k < i; k++) {
            phantom = objects[k] >> 1;
            dp = dims[phantom] + dh;
            d = 0;
            
            for (int l = 0; l < DIM; l++) {
              dx = positions[phantom + l * n] -
                  positions[home + l * n];
              d += dx * dx;
            }
            
            printf("(%d, %d): %f / %f\n", home, phantom, dp * dp, d);
            
            // if collision
            if (d < dp * dp) {
              collisions++;
            }
          }
        }
      }
      
      // if we're already past the cells assigned to this thread, we're done
      if (i > thread_end || i >= m) {
        break;
      }
      
      // the first thread starts immediately; the others wait until a change
      if (i != thread_start || !blockIdx.x && !threadIdx.x) {
        // reset counters for new cell
        h = 0;
        p = 0;
        start = i;
        last = cells[i];
      }
<<<<<<< Updated upstream
=======
      
      last = cells[i] >> 1;
>>>>>>> Stashed changes
    }
    
    // only process collisions that are not handled by a previous thread
    if (start + 1) {
      // increment home or phantom cell counter as appropriate
      if (objects[i] & 0x01) {
        h++;
      } else {
<<<<<<< Updated upstream
=======
        // increment phantom cells
>>>>>>> Stashed changes
        p++;
      }
    }
    
    i++;
  }
  
  t[threadIdx.x] = collisions;
  dSumReduce(t, collision_count);
}

__global__ void InitCellKernel(uint32_t *cells, uint32_t *objects, 
                               float *positions, float *dims, unsigned int n,
                               float cell_dim, unsigned int *cell_count) {
  extern __shared__ unsigned int t[];
  unsigned int count = 0;
  
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
      
      // we're only using the first dimension (assume circular objects)
      a = dims[i];
      sides <<= 2;
      
      // keep track of which side of the center, if any, the object overlaps
      if (x < a) {
        sides |= 3;
      } else if (cell_dim - x < a) {
        sides |= 1;
      }
    }
    
    // bit 0 unset indicates home cell
    cells[h] = hash << 1 | 0x00;
    objects[h] = i << 1 | 0x01;
    count++;
    
    // find phantom cells in the Moore neighborhood
    for (int j = 0; j < DIM_3; j++) {
      // skip the home (center) cell since it's already been added
      if (j == DIM_3 / 2) {
        continue;
      }
      
      // run through the components of each potential side cell
      q = j;
      hash = 0;
      
      for (int k = 0; k < DIM; k++) {
        r = q % 3 - 1;
        x = positions[n * k + i];
        
        // skip this cell if the object is on the wrong side
        if (r && (sides >> (DIM - k - 1) * 2 & 0x03 ^ r) & 0x03 ||
            x + r * cell_dim < 0 || x + r * cell_dim >= 1) {
          hash = UINT32_MAX;
          break;
        }
        
        // cell ID of the neighboring cell
        hash = hash << 8 | (uint32_t) (x / cell_dim) + r;
        q /= 3;
      }
      
      // only add this cell to the list if there's potential overlap
      if (hash != UINT32_MAX) {
        // count total number of cells occupied
        count++;
        h++;
        
        cells[h] = hash << 1 | 0x01;
        
        // bit 0 set indicates phantom cell
        objects[h] = i << 1 | 0x00;
        
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
  
  // perform reduction to count number of cells occupied
  t[threadIdx.x] = count;
  dSumReduce(t, cell_count);
}

__global__ void PrefixSumKernel(uint32_t *values, unsigned int n) {
  extern __shared__ uint32_t s[];
  
  for (int i = 0; i < n; i++) {
    s[i] = values[i];
  }
  
  dPrefixSum(s, n);
  
  for (int i = 0; i < n; i++) {
    values[i] = s[i];
  }
}

__global__ void RadixOrderKernel(uint32_t *keys_in, uint32_t *values_in,
                                 uint32_t *keys_out, uint32_t *values_out,
                                 uint32_t *radices, uint32_t *radix_sums,
                                 unsigned int n, unsigned int cells_per_group,
                                 int shift) {
  extern __shared__ uint32_t s[];
  uint32_t *t = s + NUM_RADICES;
  int group = threadIdx.x / THREADS_PER_GROUP;
  int group_start = (blockIdx.x * GROUPS_PER_BLOCK + group) * cells_per_group;
  int group_end = group_start + cells_per_group;
  uint32_t k;
  
  // initialize shared memory
  for (int i = threadIdx.x; i < NUM_RADICES; i += blockDim.x) {
    s[i] = radix_sums[i];
    
    // copy the last element in each prefix-sum to a separate array
    if (!((i + 1) % (NUM_RADICES / NUM_BLOCKS))) {
      t[i / (NUM_RADICES / NUM_BLOCKS)] = s[i];
    }
  }
  
  __syncthreads();
  
  // add padding to array for prefix-sum
  for (int i = threadIdx.x + NUM_BLOCKS; i < PADDED_BLOCKS; i += blockDim.x) {
    t[i] = 0;
  }
  
  __syncthreads();
  
  // calculate prefix-sum on radix counters
  dPrefixSum(t, PADDED_BLOCKS);
  __syncthreads();
  
  // add offsets to prefix-sum values
  for (int i = threadIdx.x; i < NUM_RADICES; i += blockDim.x) {
    s[i] += t[i / (NUM_RADICES / NUM_BLOCKS)];
  }
  
  __syncthreads();
  
  // add offsets to radix counters
  for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * NUM_RADICES; i +=
      blockDim.x) {
    t[i] = radices[(i / GROUPS_PER_BLOCK * NUM_BLOCKS + blockIdx.x) *
        GROUPS_PER_BLOCK + i % GROUPS_PER_BLOCK] + s[i / GROUPS_PER_BLOCK];
  }
  
  __syncthreads();
  
  // rearrange key-value pairs
  for (int i = group_start + threadIdx.x % THREADS_PER_GROUP; i < group_end &&
      i < n; i += THREADS_PER_GROUP) {
    // need only avoid bank conflicts by group
    k = (keys_in[i] >> shift & NUM_RADICES - 1) * GROUPS_PER_BLOCK + group;
    
    // write key-value pairs sequentially by thread in the thread group
    for (int j = 0; j < THREADS_PER_GROUP; j++) {
      if (threadIdx.x % THREADS_PER_GROUP == j) {
        keys_out[t[k]] = keys_in[i];
        values_out[t[k]] = values_in[i];
        t[k]++;
      }
    }
  }
}

__global__ void RadixSumKernel(uint32_t *radices, uint32_t *radix_sums) {
  extern __shared__ uint32_t s[];
  uint32_t total;
  uint32_t left = 0;
  uint32_t *radix = radices + blockIdx.x * NUM_RADICES * GROUPS_PER_BLOCK;
  
  for (int j = 0; j < NUM_RADICES / NUM_BLOCKS; j++) {
    // initialize shared memory
    for (int i = threadIdx.x; i < NUM_BLOCKS * GROUPS_PER_BLOCK; i +=
        blockDim.x) {
      s[i] = radix[i];
    }
    
    __syncthreads();
    
    // add padding to array for prefix-sum
    for (int i = threadIdx.x + NUM_BLOCKS * GROUPS_PER_BLOCK; i <
        PADDED_GROUPS; i += blockDim.x) {
      s[i] = 0;
    }
    
    __syncthreads();
    
    if (!threadIdx.x) {
      total = s[PADDED_GROUPS - 1];
    }
    
    // calculate prefix-sum on radix counters
    dPrefixSum(s, PADDED_GROUPS);
    __syncthreads();
    
    // copy to global memory
    for (int i = threadIdx.x; i < NUM_BLOCKS * GROUPS_PER_BLOCK; i +=
        blockDim.x) {
      radix[i] = s[i];
    }
    
    __syncthreads();
    
    // calculate total sum and copy to global memory
    if (!threadIdx.x) {
      total += s[PADDED_GROUPS - 1];
      
      // calculate prefix-sum on local radices
      radix_sums[blockIdx.x * NUM_RADICES / NUM_BLOCKS + j] = left;
      total += left;
      left = total;
    }
    
    // move to next radix
    radix += NUM_BLOCKS * GROUPS_PER_BLOCK;
  }
}

__global__ void RadixTabulateKernel(uint32_t *keys, uint32_t *radices,
                                    unsigned int n,
                                    unsigned int cells_per_group, int shift) {
  extern __shared__ uint32_t s[];
  int group = threadIdx.x / THREADS_PER_GROUP;
  int group_start = (blockIdx.x * GROUPS_PER_BLOCK + group) * cells_per_group;
  int group_end = group_start + cells_per_group;
  uint32_t k;
  
  // initialize shared memory
  for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * NUM_RADICES; i +=
      blockDim.x) {
    s[i] = 0;
  }
  
  __syncthreads();
  
  // count instances of each radix
  for (int i = group_start + threadIdx.x % THREADS_PER_GROUP; i < group_end &&
      i < n; i += THREADS_PER_GROUP) {
    // need only avoid bank conflicts by group
    k = (keys[i] >> shift & NUM_RADICES - 1) * GROUPS_PER_BLOCK + group;
    
    // increment radix counters sequentially by thread in the thread group
    for (int j = 0; j < THREADS_PER_GROUP; j++) {
      if (threadIdx.x % THREADS_PER_GROUP == j) {
        s[k]++;
      }
    }
  }
  
  __syncthreads();
  
  // copy to global memory
  for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * NUM_RADICES; i +=
      blockDim.x) {
    radices[(i / GROUPS_PER_BLOCK * NUM_BLOCKS + blockIdx.x) *
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
 * @brief Performs narrow-phase collision detection on a sorted cell array.
 * @param cells array of cells sorted by cudaSortCells.
 * @param objects array of corresponding objects.
 * @param positions array of object positions.
 * @param velocities array of object velocities.
 * @param dims array of object sizes.
 * @param num_objects the number of objects in the arrays.
 * @param num_cells the number of occupied cells.
 * @param temp temporary global memory variable of length at least one.
 * @return The number of collisions encountered.
 */
unsigned int cudaCellCollide(uint32_t *cells, uint32_t *objects,
                             float *positions, float *velocities, float *dims,
                             unsigned int num_objects, unsigned int num_cells,
                             unsigned int *temp, unsigned int num_blocks,
                             unsigned int threads_per_block) {
  unsigned int cells_per_thread = (num_cells - 1) / num_blocks /
      threads_per_block + 1;
  unsigned int collision_count;
  
  cudaMemset(temp, 0, sizeof(unsigned int));
  cellCollideKernel<<<num_blocks, threads_per_block,
                      threads_per_block * sizeof(unsigned int)>>>(
      cells, objects, positions, velocities, dims, num_objects, num_cells,
      cells_per_thread, temp);
  cudaMemcpy(&collision_count, temp, sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  return collision_count;
}

/**
 * @brief Constructs cell array.
 * @param cells array of cells.
 * @param objects array of corresponding objects.
 * @param positions array of object positions.
 * @param dims array of object sizes
 * @param num_objects the number of objects to process.
 * @param cell_dim the size of each cell in any dimension.
 * @param temp temporary global memory variable of length at least one.
 * @return The number of cells associations occupied
 */
unsigned int cudaInitCells(uint32_t *cells, uint32_t *objects,
                           float *positions, float *dims,
                           unsigned int num_objects, float cell_dim,
                           unsigned int *temp, unsigned int num_blocks,
                           unsigned int threads_per_block) {
  unsigned int cell_count;
  
  cudaMemset(temp, 0, sizeof(unsigned int));
  InitCellKernel<<<num_blocks, threads_per_block,
                   threads_per_block * sizeof(unsigned int)>>>(
      cells, objects, positions, dims, num_objects, cell_dim, temp);
  cudaMemcpy(&cell_count, temp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  return cell_count;
}

/**
 * @brief Randomly generates object properties.
 * @param positions array of positions.
 * @param velocities array of velocities.
 * @param dims array of dimensions.
 * @param num_objects the number of objects to generate.
 * @param max_speed the maximum possible speed in any dimension.
 * @param max_dim the maximum size in any dimension.
 */
void cudaInitObjects(float *positions, float *velocities, float *dims,
                     unsigned int num_objects, float max_speed, float max_dim,
                     unsigned int num_blocks, unsigned int threads_per_block) {
  curandGenerator_t generator;
  
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  
  // randomly generate positions ranging from (0, 0) to (1, 1)
  curandGenerateUniform(generator, positions, num_objects * DIM);
  
  // randomly generate speeds ranging from -max_speed to max_speed
  curandGenerateUniform(generator, velocities, num_objects * DIM);
  ScaleOffsetKernel<<<num_blocks, threads_per_block>>>(
      velocities, max_speed * 2, -max_speed, num_objects * DIM);
  
  // randomly generate sizes ranging from 0 to max_dim
  curandGenerateUniform(generator, dims, num_objects * DIM);
  ScaleOffsetKernel<<<num_blocks, threads_per_block>>>(
      dims, max_dim / 2, 0, num_objects * DIM);
}

/**
 * @brief Parallelized prefix-sum algorithm.
 * @param values array of values.
 * @param n the number of values to process.
 */
void cudaPrefixSum(uint32_t *values, unsigned int n) {
  PrefixSumKernel<<<1, GROUPS_PER_BLOCK * THREADS_PER_GROUP,
                    n * sizeof(uint32_t)>>>(
      values, n);
}

/**
 * @brief Radix sorts an array of objects using occupations as keys.
 * @param cells the input array of cells.
 * @param objects the input array of objects.
 * @param cells_temp sorted array of cells.
 * @param objects_temp array of objects sorted by corresponding cells.
 * @param radices working array to hold radix data.
 * @param radix_sums working array to hold radix prefix sums.
 * @param num_objects the number of objects included.
 */
void cudaSortCells(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp,
                   uint32_t *objects_temp, uint32_t *radices,
                   uint32_t *radix_sums, unsigned int num_objects) {
  unsigned int cells_per_group = (num_objects * DIM_2 - 1) / NUM_BLOCKS /
      GROUPS_PER_BLOCK + 1;
  uint32_t *cells_swap;
  uint32_t *objects_swap;
  
  // stable sort, works on bits of increasing level
  for (int i = 0; i < 32; i += L) {
    RadixTabulateKernel<<<NUM_BLOCKS, GROUPS_PER_BLOCK * THREADS_PER_GROUP,
                          GROUPS_PER_BLOCK * NUM_RADICES * sizeof(uint32_t)>>>(
        cells, radices, num_objects * DIM_2, cells_per_group, i);
    RadixSumKernel<<<NUM_BLOCKS, GROUPS_PER_BLOCK * THREADS_PER_GROUP,
                     PADDED_GROUPS * sizeof(uint32_t)>>>(
        radices, radix_sums);
    RadixOrderKernel<<<NUM_BLOCKS, GROUPS_PER_BLOCK * THREADS_PER_GROUP,
                       NUM_RADICES * sizeof(uint32_t) + GROUPS_PER_BLOCK *
                       NUM_RADICES * sizeof(uint32_t)>>>(
        cells, objects, cells_temp, objects_temp, radices, radix_sums,
        num_objects * DIM_2, cells_per_group, i);
    
    // cells sorted up to this bit are in cells_temp; swap for the next pass
    cells_swap = cells;
    cells = cells_temp;
    cells_temp = cells_swap;
    objects_swap = objects;
    objects = objects_temp;
    objects_temp = objects_swap;
  }
}
