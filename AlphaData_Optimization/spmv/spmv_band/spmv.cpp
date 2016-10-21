/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include "spmv.h"
#include <string.h>

#include "ap_int.h"

typedef ap_int<128> uint128_t;
typedef ap_int<256> uint256_t;
typedef ap_int<512> uint512_t;

extern "C" {

#define WIDTH_FACTOR_DOUBLE 8
#define WIDTH_FACTOR_SHORT 8

#define TYPE_DOUBLE uint512_t
#define TYPE_SHORT uint128_t

#define ROWS_PER_TILE 128
#define UNROLL_FACTOR 16

void ellpack(TYPE* nzval, short* cols, TYPE* vec, TYPE* out)
{
#pragma HLS INLINE off
    int i, j;
    TYPE Si, sum;

    for (i=0; i<ROWS_PER_TILE/UNROLL_FACTOR; i++) {
    #pragma HLS PIPELINE
	out[i] = 0.0;
    }

    ellpack_2 : for (j=0; j<L; j++) {
    #pragma HLS PIPELINE
        ellpack_1 : for (i=0; i<ROWS_PER_TILE/UNROLL_FACTOR; i++) {
	#pragma HLS UNROLL
            out[i] = out[i] + nzval[j + i*L] * vec[cols[j + i*L]];
        }
    }
}

void load_nzval(TYPE_DOUBLE* nzval, TYPE_DOUBLE local_nzval[][L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_DOUBLE], int flag) {
#pragma HLS INLINE off
  int j;
  if (flag) {
      for (j=0; j<UNROLL_FACTOR; j++) {
          memcpy(local_nzval[j], nzval + j*ROWS_PER_TILE/UNROLL_FACTOR*L/WIDTH_FACTOR_DOUBLE, sizeof(TYPE)*ROWS_PER_TILE*L/UNROLL_FACTOR);
      }
  }
}

void load_cols(TYPE_SHORT* cols, TYPE_SHORT local_cols[][L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_SHORT], int flag) {
#pragma HLS INLINE off
  int j;
  if (flag) {
      for (j=0; j<UNROLL_FACTOR; j++) {
	  memcpy(local_cols[j], cols + j*ROWS_PER_TILE/UNROLL_FACTOR*L/WIDTH_FACTOR_SHORT, sizeof(short)*ROWS_PER_TILE*L/UNROLL_FACTOR);
      }
  }
}

void ellpack_wrapper(TYPE_DOUBLE* nzval, TYPE_SHORT* cols, TYPE* vec, TYPE* out)
{
  TYPE local_nzval[L*ROWS_PER_TILE/UNROLL_FACTOR];
  short local_cols[L*ROWS_PER_TILE/UNROLL_FACTOR];
  int i, j;
  TYPE_DOUBLE tmp_double;
  TYPE_SHORT tmp_short;
  uint64_t int_to_double;
  for (i=0; i<L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_DOUBLE; i++) {
  #pragma HLS PIPELINE
    tmp_double = nzval[i];
    for (j=0; j<WIDTH_FACTOR_DOUBLE; j++) {
      int_to_double = tmp_double.range(63, 0);
      local_nzval[i*WIDTH_FACTOR_DOUBLE+j] = *((double *)(&int_to_double));
      tmp_double = tmp_double >> 64;
    }
  }
  for (i=0; i<L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_SHORT; i++) {
  #pragma HLS PIPELINE
    tmp_short = cols[i];
    for (j=0; j<WIDTH_FACTOR_SHORT; j++) {
      local_cols[i*WIDTH_FACTOR_SHORT+j] = tmp_short.range(15, 0);
      tmp_short = tmp_short >> 16;
    }
  }

  ellpack(local_nzval, local_cols, vec, out);

}

void buffer_compute(TYPE_DOUBLE local_nzval[][L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_DOUBLE], TYPE_SHORT local_cols[][L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_SHORT], 
		    TYPE local_vec[][N], TYPE local_out[][ROWS_PER_TILE/UNROLL_FACTOR], int flag, TYPE* out) {
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j=0; j<UNROLL_FACTOR; j++) {
    #pragma HLS UNROLL
        ellpack_wrapper(local_nzval[j], local_cols[j], local_vec[j], local_out[j]);
    }
    for (j=0; j<UNROLL_FACTOR; j++) {
	memcpy(out + j*ROWS_PER_TILE/UNROLL_FACTOR, local_out[j], sizeof(TYPE)*ROWS_PER_TILE/UNROLL_FACTOR);
    }
  }
}


void workload(TYPE_DOUBLE* nzval, TYPE_SHORT* cols, TYPE* vec, TYPE* out) {
#pragma HLS INTERFACE m_axi port=nzval offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=cols offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=vec offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=nzval bundle=control
#pragma HLS INTERFACE s_axilite port=cols bundle=control
#pragma HLS INTERFACE s_axilite port=vec bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int num_tiles = N/ROWS_PER_TILE;
	int i,j,k;

	TYPE_DOUBLE local_nzval_x[UNROLL_FACTOR][L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_DOUBLE];
	#pragma HLS ARRAY_PARTITION variable=local_nzval_x dim=1 complete
	TYPE_SHORT local_cols_x[UNROLL_FACTOR][L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_SHORT];
	#pragma HLS ARRAY_PARTITION variable=local_cols_x dim=1 complete

	TYPE_DOUBLE local_nzval_y[UNROLL_FACTOR][L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_DOUBLE];
	#pragma HLS ARRAY_PARTITION variable=local_nzval_y dim=1 complete
	TYPE_SHORT local_cols_y[UNROLL_FACTOR][L*ROWS_PER_TILE/UNROLL_FACTOR/WIDTH_FACTOR_SHORT];
	#pragma HLS ARRAY_PARTITION variable=local_cols_y dim=1 complete

	TYPE local_vec[UNROLL_FACTOR][N];
	#pragma HLS ARRAY_PARTITION variable=local_vec dim=1 complete
	memcpy(local_vec[0], vec, N*sizeof(TYPE));
	for(i=0; i<N; i++) {
        #pragma HLS PIPELINE
	    for(j=1; j<UNROLL_FACTOR; j++) {
	    #pragma HLS UNROLL
	        local_vec[j][i] = local_vec[0][i];
	    }
	} 

	TYPE local_out[UNROLL_FACTOR][ROWS_PER_TILE/UNROLL_FACTOR];
	#pragma HLS ARRAY_PARTITION variable=local_out dim=1 complete

	int load_flag, compute_flag;
	for (i=0; i<num_tiles+1; i++) {
	    load_flag = i >= 0 && i < num_tiles;
	    compute_flag = i > 0 && i <= num_tiles;
	    if (i % 2 == 0) {
		load_nzval(nzval + i*ROWS_PER_TILE*L/WIDTH_FACTOR_DOUBLE, local_nzval_x, load_flag);
		load_cols(cols + i*ROWS_PER_TILE*L/WIDTH_FACTOR_SHORT, local_cols_x, load_flag);
		buffer_compute(local_nzval_y, local_cols_y, local_vec, local_out, compute_flag, out + (i-1)*ROWS_PER_TILE);
	    }
	    else {
		load_nzval(nzval + i*ROWS_PER_TILE*L/WIDTH_FACTOR_DOUBLE, local_nzval_y, load_flag);
		load_cols(cols + i*ROWS_PER_TILE*L/WIDTH_FACTOR_SHORT, local_cols_y, load_flag);
		buffer_compute(local_nzval_x, local_cols_x, local_vec, local_out, compute_flag, out + (i-1)*ROWS_PER_TILE);
	    }
	}

	return;
}

}
