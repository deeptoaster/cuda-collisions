#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <cuda_runtime.h>
#include "collisions.cuh"

#define NUM_OBJECTS 65536
#define MAX_SPEED 0.5
#define MAX_DIM 0.25
#define NUM_DISPLAY 4

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

unsigned int num_blocks = 100;
unsigned int threads_per_block = 512;
unsigned int object_size = NUM_OBJECTS * DIM * sizeof(float);
unsigned int cell_size = NUM_OBJECTS * DIM_2 * sizeof(uint32_t);
float *positions = (float *) malloc(object_size);
float *velocities = (float *) malloc(object_size);
float *dims = (float *) malloc(object_size);
float *d_positions;
float *d_velocities;
float *d_dims;
uint32_t *cells = (uint32_t *) malloc(cell_size);
uint32_t *objects = (uint32_t *) malloc(cell_size);
uint32_t *radices = (uint32_t *) malloc(NUM_BLOCKS * GROUPS_PER_BLOCK *
                                        NUM_RADICES * sizeof(uint32_t));
uint32_t *radix_sums = (uint32_t *) malloc(NUM_RADICES * sizeof(uint32_t));
uint32_t *d_cells_in;
uint32_t *d_cells_out;
uint32_t *d_objects_in;
uint32_t *d_objects_out;
uint32_t *d_radices;
uint32_t *d_radix_sums;

inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %4d\n", cudaGetErrorString(code), file,
        line);
    exit(code);
  }
}

void TestPrefixSum(int n) {
  uint32_t *arr = (uint32_t *) malloc(n * sizeof(uint32_t));
  uint32_t *d_arr;
  cudaMalloc((void **) &d_arr, n * sizeof(uint32_t));
  
  for (int i = 0; i < n; i++) {
    arr[i] = 1;
  }
  
  cudaMemcpy(d_arr, arr, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaPrefixSum(d_arr, n);
  cudaMemcpy(arr, d_arr, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < n / 2; i++) {
    printf("%d\t%d\n", arr[i * 2], arr[i * 2 + 1]);
  }
}

void TestSortCells() {
  cudaSortCells(d_cells_in, d_objects_in, d_cells_out, d_objects_out,
                d_radices, d_radix_sums, NUM_OBJECTS);
  
  for (int i = 0; i < NUM_DISPLAY; i++) {
    printf("% f % f % f % f\t% f % f % f % f\t% f % f % f % f\n",
           positions[i * 4], positions[i * 4 + 1], positions[i * 4 + 2], 
           positions[i * 4 + 3], positions[i * 4 + NUM_OBJECTS],
           positions[i * 4 + 1 + NUM_OBJECTS],
           positions[i * 4 + 2 + NUM_OBJECTS], 
           positions[i * 4 + 3 + NUM_OBJECTS], dims[i * 4], dims[i * 4 + 1],
           dims[i * 4 + 2], dims[i * 4 + 3]);
  }
  
  cudaMemcpy(cells, d_cells_in, cell_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(objects, d_objects_in, cell_size, cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < NUM_DISPLAY; i++) {
    printf("%8x %8x %8x %8x\t%4d %4d %4d %4d\n", cells[i * 4],
           cells[i * 4 + 1], cells[i * 4 + 2], cells[i * 4 + 3],
           objects[i * 4], objects[i * 4 + 1], objects[i * 4 + 2],
           objects[i * 4 + 3]);
  }
  
  cudaMemcpy(cells, d_cells_out, cell_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(objects, d_objects_out, cell_size, cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < NUM_DISPLAY; i++) {
    printf("%8x %8x %8x %8x\t%4d %4d %4d %4d\n", cells[i * 4],
           cells[i * 4 + 1], cells[i * 4 + 2], cells[i * 4 + 3],
           objects[i * 4], objects[i * 4 + 1], objects[i * 4 + 2],
           objects[i * 4 + 3]);
  }
  
  cudaMemcpy(radices, d_radices, NUM_BLOCKS * GROUPS_PER_BLOCK * NUM_RADICES *
             sizeof(uint32_t), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < NUM_DISPLAY; i++) {
    printf("%4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d\n",
           radices[i * 16], radices[i * 16 + 1], radices[i * 16 + 2],
           radices[i * 16 + 3], radices[i * 16 + 4], radices[i * 16 + 5],
           radices[i * 16 + 6], radices[i * 16 + 7], radices[i * 16 + 8],
           radices[i * 16 + 9], radices[i * 16 + 10], radices[i * 16 + 11],
           radices[i * 16 + 12], radices[i * 16 + 13], radices[i * 16 + 14],
           radices[i * 16 + 15]);
  }
  
  cudaMemcpy(radix_sums, d_radix_sums, NUM_RADICES *
             sizeof(uint32_t), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < NUM_DISPLAY; i++) {
    printf("%4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d %4d\n",
           radix_sums[i * 16], radix_sums[i * 16 + 1], radix_sums[i * 16 + 2],
           radix_sums[i * 16 + 3], radix_sums[i * 16 + 4], radix_sums[i * 16 + 5],
           radix_sums[i * 16 + 6], radix_sums[i * 16 + 7], radix_sums[i * 16 + 8],
           radix_sums[i * 16 + 9], radix_sums[i * 16 + 10], radix_sums[i * 16 + 11],
           radix_sums[i * 16 + 12], radix_sums[i * 16 + 13], radix_sums[i * 16 + 14],
           radix_sums[i * 16 + 15]);
  }
}

int main(int argc, char *argv[]) {
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
  cudaMemset(d_cells_out, 0, cell_size);
  cudaMemset(d_objects_out, 0, cell_size);
  cudaMemset(d_radices, 0, NUM_BLOCKS * GROUPS_PER_BLOCK * NUM_RADICES *
             sizeof(uint32_t));
  cudaMemset(d_radix_sums, 0, NUM_RADICES * sizeof(uint32_t));
  cudaInitObjects(d_positions, d_velocities, d_dims, NUM_OBJECTS, MAX_SPEED, 
                  MAX_DIM, num_blocks, threads_per_block);
  cudaInitCells(d_cells_in, d_objects_in, d_positions, d_dims, NUM_OBJECTS,
                MAX_DIM, num_blocks, threads_per_block);
  cudaMemcpy(positions, d_positions, object_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(velocities, d_velocities, object_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(dims, d_dims, object_size, cudaMemcpyDeviceToHost);
  //~ TestPrefixSum(256);
  TestSortCells();
  gpuErrChk(cudaGetLastError());
}
