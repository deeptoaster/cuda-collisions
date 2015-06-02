#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <cuda_runtime.h>
#include "collisions.cuh"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
        line);
    exit(code);
  }
}

unsigned int num_blocks = 100;
unsigned int threads_per_block = 512;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf(
        "Usage: %s NUMOBJECTS MAXSPEED MAXDIM [NUMBLOCKS [THREADSPERBLOCK]]\n",
        argv[0]);
    return -1;
  }
  
  unsigned int num_objects = atoi(argv[1]);
  float max_speed = atoi(argv[2]);
  float max_dim = atoi(argv[3]);
  
  if (argc >= 5) {
    num_blocks = atoi(argv[4]);
  }
  
  if (argc >= 6) {
    threads_per_block = atoi(argv[5]);
  }
  
  unsigned int object_size = (num_objects - 1) / threads_per_block + 1;
  
  if (object_size < num_blocks) {
    num_blocks = object_size;
  }
  
  object_size = num_objects * DIM * sizeof(float);
  
  unsigned int cell_size = num_objects * DIM_2 * sizeof(uint32_t);
  float *positions = (float *) malloc(object_size);
  float *velocities = (float *) malloc(object_size);
  float *dims = (float *) malloc(object_size);
  float *d_positions;
  float *d_velocities;
  float *d_dims;
  uint32_t *d_cells_in;
  uint32_t *d_cells_out;
  uint32_t *d_objects_in;
  uint32_t *d_objects_out;
  uint32_t *d_radices;
  uint32_t *d_radix_sums;
  
  cudaMalloc((void **) &d_positions, object_size);
  cudaMalloc((void **) &d_velocities, object_size);
  cudaMalloc((void **) &d_dims, object_size);
  cudaMalloc((void **) &d_cells_in, cell_size);
  cudaMalloc((void **) &d_cells_out, cell_size);
  cudaMalloc((void **) &d_objects_in, cell_size);
  cudaMalloc((void **) &d_objects_out, cell_size);
  cudaMalloc((void **) &d_radices, NUM_BLOCKS * GROUPS_PER_BLOCK *
             NUM_RADICES * sizeof(uint32_t));
  cudaMalloc((void **) &d_radix_sums, NUM_RADICES * sizeof(uint32_t));
  cudaInitObjects(d_positions, d_velocities, d_dims, num_objects, max_speed, 
                  max_dim, num_blocks, threads_per_block);
  cudaInitCells(d_cells_in, d_objects_in, d_positions, d_dims, num_objects,
                max_dim, num_blocks, threads_per_block);
  cudaSortCells(d_cells_in, d_objects_in, d_cells_out, d_objects_out,
                d_radices, d_radix_sums, num_objects);
  cudaMemcpy(positions, d_positions, object_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(velocities, d_velocities, object_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(dims, d_dims, object_size, cudaMemcpyDeviceToHost);
  
  
  
  cudaFree(d_positions);
  cudaFree(d_velocities);
  cudaFree(d_dims);
  cudaFree(d_cells_in);
  cudaFree(d_cells_out);
  cudaFree(d_objects_in);
  cudaFree(d_objects_out);
  cudaFree(d_radices);
  cudaFree(d_radix_sums);
  free(positions);
  free(velocities);
  free(dims);
  
  gpuErrChk(cudaGetLastError());
  
  return 0;
}
