//Standard Libraries
#ifndef LINEARNODE_H
#define LINEARNODE_H
#include <stdio.h>
#include <stdlib.h>

//Define compute data type
#define TYPE double


#define MatrixSize 1024 
//Specify row/column sizes
#define row_size MatrixSize
#define col_size MatrixSize 
#define N row_size*col_size

#define T 128
#define tile_size T

#define unroll_size 64
//#define tile_size 64
//#define T tile_size
#define debug 0
//Define the input range to operate over
#define MIN 0.
#define MAX 1.0

//Set number of iterations to execute
#define MAX_ITERATION 1

//void workload(TYPE m1[N], TYPE m2[N], TYPE prod[N]);
//void compute(int flag, TYPE local_m1[T][T], TYPE local_m2[T][T], TYPE local_prod[T][T]);
//void load(int flag, int i, int j, int k, TYPE local_m1[T][T], TYPE local_m2[T][T], TYPE m1[N], TYPE m2[N]);
//void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  TYPE m1[N];
  TYPE m2[N];
  TYPE prod[N];
};

#endif
