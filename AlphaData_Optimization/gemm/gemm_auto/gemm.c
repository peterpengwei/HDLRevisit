#include "gemm.h"

#define UNROLL 2
#define CYCLIC 32

#define ELE_PER_UNROLL ((MatrixSize+UNROLL-1)/UNROLL)

void load(int flag, TYPE* dst, TYPE* src, int num_rows) {
#pragma HLS inline off
  if (flag) {
    memcpy(dst, src, sizeof(TYPE)*MatrixSize*num_rows);
  }
}

void compute(int flag, TYPE rows[UNROLL][MatrixSize], TYPE col[MatrixSize][CYCLIC], TYPE* prod, int num_rows) {
#pragma HLS inline off
  if (flag) {
    TYPE local_prod[UNROLL][CYCLIC];
#pragma HLS array_partition variable=local_prod dim=1 complete
    int i,j,k;

    for (j=0; j<CYCLIC; j++) {
#pragma HLS pipeline
    for (i=0; i<UNROLL; i++) {
#pragma HLS unroll
      local_prod[i][j] = 0.0;
    }
    }

    for (j=0; j<MatrixSize; j++) {
      for (k=0; k<CYCLIC; k++) {
#pragma HLS pipeline
#pragma HLS dependence variable=local_prod inter false
      for (i=0; i<UNROLL; i++) {
#pragma HLS unroll
	local_prod[i][k] += rows[i][j]*col[j][k];
      }
      }
    }

    for (i=0; i<UNROLL; i++) {
    for (j=0; j<CYCLIC; j++) {
#pragma HLS pipeline
      prod[i*MatrixSize+j] = local_prod[i][j];
    }
    }
  }
}

void workload(TYPE m1[N], TYPE m2[N], TYPE prod[N]){
#pragma HLS INTERFACE m_axi port=m1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=m2 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=prod offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=m1 bundle=control
#pragma HLS INTERFACE s_axilite port=m2 bundle=control
#pragma HLS INTERFACE s_axilite port=prod bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    TYPE local_col[MatrixSize][CYCLIC];
//#pragma HLS array_partition variable=local_col dim=2 complete
    TYPE local_rows_x[UNROLL][MatrixSize];
#pragma HLS array_partition variable=local_rows_x dim=1 complete
    TYPE local_rows_y[UNROLL][MatrixSize];
#pragma HLS array_partition variable=local_rows_y dim=1 complete
//    TYPE local_out[ELE_PER_UNROLL][UNROLL];
//#pragma HLS array_partition variable=local_out dim=2 complete 

    int i,j,k;
    for (i=0; i<MatrixSize; i+=CYCLIC) { // i is col index
      for (j=0; j<MatrixSize; j++) { // j here is row index
      for (k=0; k<CYCLIC; k++) {
#pragma HLS pipeline
        local_col[j][k] = m2[j*MatrixSize+i+k];
      }
      }
      int num_batches = (MatrixSize + UNROLL - 1)/UNROLL;
      int tail_size = num_batches % UNROLL;
      if (tail_size == 0) tail_size = UNROLL;
      for (j=0; j<num_batches+1; j++) {
        int load_flag = j < num_batches;
	int compute_flag = j > 0;
	int load_rows = j == num_batches-1? tail_size:UNROLL;
	int compute_rows = j == num_batches? tail_size:UNROLL;
	if (j % 2 == 0) {
	  load(load_flag, local_rows_x, m1+j*MatrixSize*UNROLL, load_rows);
	  compute(compute_flag, local_rows_y, local_col, prod+(j-1)*MatrixSize+i, compute_rows);
	}
	else {
	  load(load_flag, local_rows_y, m1+j*MatrixSize*UNROLL, load_rows);
	  compute(compute_flag, local_rows_x, local_col, prod+(j-1)*MatrixSize+i, compute_rows);
	}
      }
//      for (j=0; j<MatrixSize; j++) {
//#pragma HLS pipeline
//        prod[j*MatrixSize+i] = local_out[j/UNROLL][j%UNROLL];
//      }
    }
    return;
}
