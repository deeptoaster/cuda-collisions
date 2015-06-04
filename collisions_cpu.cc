#include <cstdio>

#include "collisions.h"
#include "collisions_cpu.h"

/**
 * @brief Counts the number of collisions among objects.
 * @param positions array of positions.
 * @param velocities array of velocities.
 * @param dims array of dimensions.
 * @param num_objects the number of objects in the arrays.
 * @return The number of collisions encountered
 */
unsigned int CellCollide(float *positions, float *velocities, float *dims,
                         unsigned int num_objects) {
  unsigned int collisions = 0;
  float dh;
  float dp;
  float dx;
  float d;
  
  for (int j = 0; j < num_objects; j++) {
    dh = dims[j];
    
    for (int k = j + 1; k < num_objects; k++) {
      
      // assume dims are radii of balls
      dp = dims[k] + dh;
      d = 0;
      
      for (int l = 0; l < DIM; l++) {
        dx = positions[j + l * num_objects] -
             positions[k + l * num_objects];
        d += dx * dx;
      }
      
      // if collision
      if (d < dp * dp) {
        collisions++;
      }
    }
  }
  
  return collisions;
}
