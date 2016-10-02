/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

// These constants valid for the IEEE 494 bus interconnect matrix
#define N (1 << 12)
#define L (1 << 9)

#define TYPE double

// void ellpack(TYPE nzval[N*L], short cols[N*L], TYPE vec[N], TYPE out[N]);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  TYPE nzval[N*L];
  short cols[N*L];
  TYPE vec[N];
  TYPE out[N];
};
