#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <cuda_runtime.h>
#include "collisions.cuh"

#define NUM_OBJECTS 16
#define MAX_SPEED 0.5
#define MAX_DIM 0.5
#define COLS 8

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

unsigned int num_blocks = 100;
unsigned int threads_per_block = 512;
unsigned int object_size = NUM_OBJECTS * DIM * sizeof(float);
unsigned int cell_size = NUM_OBJECTS * DIM_2 * sizeof(uint32_t);
unsigned int cell_count;
unsigned int *d_temp;
float *positions;
float *velocities;
float *dims;
float *d_positions;
float *d_velocities;
float *d_dims;
uint32_t *cells;
uint32_t *objects;
uint32_t *d_cells;
uint32_t *d_cells_temp;
uint32_t *d_objects;
uint32_t *d_objects_temp;
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

/**
 * @brief Tests cudaInitCells for cell assignment correctness.
 * @param n number of objects whose cells are checked and displayed.
 */
void TestInitCells(int n) {
  printf("Testing InitCells...\n");
  cell_count = cudaInitCells(d_cells, d_objects, d_positions, d_dims,
                             NUM_OBJECTS, MAX_DIM, d_temp, num_blocks,
                             threads_per_block);
  cudaMemcpy(cells, d_cells, cell_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(objects, d_objects, cell_size, cudaMemcpyDeviceToHost);
  
  if (NUM_OBJECTS < n) {
    n = NUM_OBJECTS;
  }
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < DIM_2; j++) {
      uint32_t cell = cells[i * DIM_2 + j];
      
      printf("%8x %8x\t", cell, objects[i * DIM_2 + j]);
      
      if (COLS < DIM_2 * 2) {
        printf("\n");
      }
      
      for (int k = 0; k < DIM; k++) {
        assert(cell == UINT32_MAX || abs((cell >> (DIM - k - 1) * 8 & 0xff) *
            MAX_DIM - positions[i + k * NUM_OBJECTS]) <
            dims[i + k * NUM_OBJECTS]);
      }
    }
    
    if (!((i + 1) % (COLS / DIM_2 / 2))) {
      printf("\n");
    } else {
      printf("\t");
    }
  }
  
  printf("\n");
}

/**
 * @brief Tests cudaInitObjects for object constraint satisfaction.
 * @param n number of objects whose properties are checked and displayed.
 */
void TestInitObjects(int n) {
  printf("Testing InitObjects...\n");
  cudaInitObjects(d_positions, d_velocities, d_dims, NUM_OBJECTS, MAX_SPEED, 
                  MAX_DIM, num_blocks, threads_per_block);
  cudaMemcpy(positions, d_positions, object_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(velocities, d_velocities, object_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(dims, d_dims, object_size, cudaMemcpyDeviceToHost);
  
  if (NUM_OBJECTS < n) {
    n = NUM_OBJECTS;
  }
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < DIM; j++) {
      printf("% f ", positions[i + j * NUM_OBJECTS]);
      assert(positions[i + j * NUM_OBJECTS] >= 0 &&
             positions[i + j * NUM_OBJECTS] < 1);
    }
    
    if (COLS < DIM * 3) {
      printf("\n");
    } else {
      printf("\t");
    }
    
    for (int j = 0; j < DIM; j++) {
      printf("% f ", velocities[i + j * NUM_OBJECTS]);
      assert(abs(velocities[i + j * NUM_OBJECTS]) < MAX_SPEED);
    }
    
    if (COLS < DIM * 3) {
      printf("\n");
    } else {
      printf("\t");
    }
    
    for (int j = 0; j < DIM; j++) {
      printf("% f ", dims[i + j * NUM_OBJECTS]);
      assert(dims[i + j * NUM_OBJECTS] < MAX_DIM / 2);
    }
    
    if (!(COLS / DIM / 3) || !((i + 1) % (COLS / DIM / 3))) {
      printf("\n");
    } else {
      printf("\t");
    }
  }
  
  printf("\n");
}

/**
 * @brief Tests cudaPrefixSum for correctness.
 * @param n number of elements on which to perform a prefix sum.
 */
void TestPrefixSum(int n) {
  printf("Testing PrefixSum...\n");
  
  uint32_t *arr = (uint32_t *) malloc(n * sizeof(uint32_t));
  uint32_t *d_arr;
  
  cudaMalloc((void **) &d_arr, n * sizeof(uint32_t));
  
  for (int i = 0; i < n; i++) {
    arr[i] = 1;
  }
  
  cudaMemcpy(d_arr, arr, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaPrefixSum(d_arr, n);
  cudaPrefixSum(d_arr, n);
  cudaMemcpy(arr, d_arr, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < n; i++) {
    printf("%d\t", arr[i]);
    
    if (!((i + 1) % COLS)) {
      printf("\n");
    }
    
    assert((int) arr[i] == i * (i - 1) / 2);
  }
  
  free(arr);
  cudaFree(d_arr);
  printf("\n");
}

/**
 * @brief Tests cudaInitCells for count and cudaSortCells for radix sort.
 * @param n when multiplied by DIM_2, the number of cells to check.
 */
void TestSortCells(int n) {
  printf("TestingSortCells...\n");
  cudaSortCells(d_cells, d_objects, d_cells_temp, d_objects_temp,
                d_radices, d_radix_sums, NUM_OBJECTS);
  cudaMemcpy(cells, d_cells, cell_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(objects, d_objects, cell_size, cudaMemcpyDeviceToHost);
  assert(cell_count && cell_count <= NUM_OBJECTS * DIM_2);
  assert(cells[cell_count] == UINT32_MAX && cells[cell_count - 1] !=
      UINT32_MAX);
  
  if (NUM_OBJECTS < n) {
    n = NUM_OBJECTS;
  }
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < DIM_2; j++) {
      uint32_t cell = cells[i * DIM_2 + j];
      
      printf("%8x %8x\t", cell, objects[i * DIM_2 + j]);
      
      if (COLS < DIM_2 * 2) {
        printf("\n");
      }
      
      for (int k = 0; k < DIM; k++) {
        assert((!i && !j) || cell >= cells[i * DIM_2 + j - 1]);
      }
    }
    
    if (!((i + 1) % (COLS / DIM_2 / 2))) {
      printf("\n");
    } else {
      printf("\t");
    }
  }
  
  printf("\n");
}

int main(int argc, char *argv[]) {
  positions = (float *) malloc(object_size);
  velocities = (float *) malloc(object_size);
  dims = (float *) malloc(object_size);
  cells = (uint32_t *) malloc(cell_size);
  objects = (uint32_t *) malloc(cell_size);
  cudaMalloc((void **) &d_temp, sizeof(unsigned int));
  cudaMalloc((void **) &d_positions, object_size);
  cudaMalloc((void **) &d_velocities, object_size);
  cudaMalloc((void **) &d_dims, object_size);
  cudaMalloc((void **) &d_cells, cell_size);
  cudaMalloc((void **) &d_cells_temp, cell_size);
  cudaMalloc((void **) &d_objects, cell_size);
  cudaMalloc((void **) &d_objects_temp, cell_size);
  cudaMalloc((void **) &d_radices, NUM_BLOCKS * GROUPS_PER_BLOCK *
             NUM_RADICES * sizeof(uint32_t));
  cudaMalloc((void **) &d_radix_sums, NUM_RADICES * sizeof(uint32_t));
  TestPrefixSum(256);
  TestInitObjects(16);
  TestInitCells(16);
  TestSortCells(16);
  gpuErrChk(cudaGetLastError());
  free(positions);
  free(velocities);
  free(dims);
  free(cells);
  free(objects);
  cudaFree(d_temp);
  cudaFree(d_positions);
  cudaFree(d_velocities);
  cudaFree(d_dims);
  cudaFree(d_cells);
  cudaFree(d_cells_temp);
  cudaFree(d_objects);
  cudaFree(d_objects_temp);
  cudaFree(d_radices);
  cudaFree(d_radix_sums);
  printf("All tests passed.\n");
  return 0;
}
