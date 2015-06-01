#include <cstdio>
#include <cstdlib>
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
  float max_velocity = atoi(argv[2]);
  float max_dim = atoi(argv[3]);
  
  if (argc >= 5) {
    num_blocks = atoi(argv[4]);
  }
  
  if (argc >= 6) {
    threads_per_block = atoi(argv[5]);
  }
  
  unsigned int size_objects = (num_objects - 1) / threads_per_block + 1;
  
  if (size_objects < num_blocks) {
    num_blocks = size_objects;
  }
  
  size_objects = num_objects * DIM * sizeof(float);
  
  float *positions = (float *) malloc(size_objects);
  float *velocities = (float *) malloc(size_objects);
  float *dims = (float *) malloc(size_objects);
  float *d_positions;
  float *d_velocities;
  float *d_dims;
  unsigned int *d_cells;
  unsigned int *d_objects;
  
  cudaMalloc((void **) &d_positions, size_objects);
  cudaMalloc((void **) &d_velocities, size_objects);
  cudaMalloc((void **) &d_dims, size_objects);
  cudaMalloc((void **) &d_cells, num_objects * DIM_2 * sizeof(unsigned int));
  cudaMalloc((void **) &d_objects, num_objects * DIM_2 * sizeof(unsigned int));
  cudaInitObjects(d_positions, d_velocities, d_dims, num_objects, max_velocity, 
                  max_dim, num_blocks, threads_per_block);
  cudaInitCells(d_cells, d_objects, d_positions, d_dims, num_objects, max_dim,
                num_blocks, threads_per_block);
  cudaMemcpy(positions, d_positions, size_objects, cudaMemcpyDeviceToHost);
  cudaMemcpy(velocities, d_velocities, size_objects, cudaMemcpyDeviceToHost);
  cudaMemcpy(dims, d_dims, size_objects, cudaMemcpyDeviceToHost);
  
  gpuErrChk(cudaGetLastError());
  
  return 0;
}
