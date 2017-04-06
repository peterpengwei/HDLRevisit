#include "stencil.h"
#include <string.h>

#define tile_size 256

#define UNROLL_R 3
#define UNROLL_C 32

#define MIN(A,B) ((A)<(B)?(A):(B))
#define ROW_IDX(A, num_cols) ((A)/(num_cols))
#define COL_IDX(A, num_cols) ((A)%(num_cols))

void stencil (TYPE orig[UNROLL_R][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C], TYPE sol[][tile_size+2], TYPE filter[], size_t row, size_t col){
#pragma HLS inline off

    int r, c, k1, k2, i, j;
    TYPE data[UNROLL_R][UNROLL_C+2];
#pragma HLS array_partition variable=data dim=1 complete
#pragma HLS array_partition variable=data dim=2 complete

    stencil_label1:for (r=0; r<row-2; r+=(UNROLL_R-2)) {
    stencil_label2:for (c=0; c<col; c+=(UNROLL_C)) {
#pragma HLS pipeline
        for (j=0; j<UNROLL_C; j++) {
#pragma HLS unroll
	    if (c+j < col-2) {
	        sol[r][c+j] = orig[r%UNROLL_R][j][r/UNROLL_R][(c+j)/UNROLL_C];
	    }	
	}
// #pragma HLS pipeline
// 	for (i=0; i<UNROLL_R; i++) {
// #pragma HLS unroll
// 	for (j=0; j<2; j++) {
// #pragma HLS unroll
// 	  data[i][j] = data[i][j+UNROLL_C];
// 	}
// 	}
// 	for (j=0; j<UNROLL_C; j++) {
// #pragma HLS unroll
// 	    if (r+2 < row && c+j < col) {
// 		switch (r % 3) {
// 		case 0:
// 		    data[0][j+2] = orig[0][j][(r+0)/UNROLL_R][(c+j)/UNROLL_C];
// 		    data[1][j+2] = orig[1][j][(r+1)/UNROLL_R][(c+j)/UNROLL_C];
// 		    data[2][j+2] = orig[2][j][(r+2)/UNROLL_R][(c+j)/UNROLL_C];
// 		    break;
// 		case 1:
// 		    data[0][j+2] = orig[1][j][(r+0)/UNROLL_R][(c+j)/UNROLL_C];
// 		    data[1][j+2] = orig[2][j][(r+1)/UNROLL_R][(c+j)/UNROLL_C];
// 		    data[2][j+2] = orig[0][j][(r+2)/UNROLL_R][(c+j)/UNROLL_C];
// 		    break;
// 		default:
// 		    data[0][j+2] = orig[2][j][(r+0)/UNROLL_R][(c+j)/UNROLL_C];
// 		    data[1][j+2] = orig[0][j][(r+1)/UNROLL_R][(c+j)/UNROLL_C];
// 		    data[2][j+2] = orig[1][j][(r+2)/UNROLL_R][(c+j)/UNROLL_C];
// 		}
// 	    }
// 	}
// 	for (j=0; j<UNROLL_C; j++) {
// #pragma HLS unroll
// 	  if (r < row-2 && c-2+j < col-2 && c-2+j >= 0) {
//              // sol[r][c-2+j] = filter[0] * data[0][j+0]
// 	     // 	           + filter[1] * data[0][j+1]
// 	     // 	           + filter[2] * data[0][j+2]
// 	     // 	           + filter[3] * data[1][j+0]
// 	     // 	           + filter[4] * data[1][j+1]
// 	     // 	           + filter[5] * data[1][j+2]
// 	     // 	           + filter[6] * data[2][j+0]
// 	     // 	           + filter[7] * data[2][j+1]
// 	     // 	           + filter[8] * data[2][j+2];
// 	     sol[r][c-2+j] = data[0][j];
// 	  }
// 	}
    }
    }
}

void load(int flag, TYPE dst[][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C], TYPE* src, size_t row, size_t col) {
#pragma HLS inline off
    if (flag) {
        for (int i=0; i<row; i++) {
	for (int j=0; j<col; j++) {
#pragma HLS pipeline
	    dst[i%UNROLL_R][j%UNROLL_C][i/UNROLL_R][j/UNROLL_C] = src[i*col_size+j];
	}
        }
    }
}

void store(int flag, TYPE* dst, TYPE src[][tile_size+2], size_t row, size_t col) {
#pragma HLS inline off
    if (flag) {
        for (int i=0; i<row; i++) {
	for (int j=0; j<col; j++) {
#pragma HLS pipeline
	    dst[i*col_size+j] = src[i][j];
	}
        }
    }
}

void compute(int flag, TYPE orig[][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C], TYPE sol[][tile_size+2], TYPE filter[f_size], size_t row, size_t col) {
#pragma HLS inline off
    if (flag && row>2 && col>2) {
        stencil(orig, sol, filter, row, col);
    }

}

void workload(TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
#pragma HLS INTERFACE m_axi port=orig offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=sol offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=filter offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=orig bundle=control
#pragma HLS INTERFACE s_axilite port=sol bundle=control
#pragma HLS INTERFACE s_axilite port=filter bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

        TYPE local_filter[f_size];
#pragma HLS array_partition variable=local_filter complete
        memcpy(local_filter, filter, sizeof(TYPE)*f_size);

        TYPE local_orig_x[UNROLL_R][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C];
#pragma HLS array_partition variable=local_orig_x dim=1 complete
#pragma HLS array_partition variable=local_orig_x dim=2 complete
        TYPE local_sol_x[tile_size+2][tile_size+2];
#pragma HLS array_partition variable=local_sol_x dim=2 cyclic factor=32
        TYPE local_orig_y[UNROLL_R][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C];
#pragma HLS array_partition variable=local_orig_y dim=1 complete
#pragma HLS array_partition variable=local_orig_y dim=2 complete
        TYPE local_sol_y[tile_size+2][tile_size+2];
#pragma HLS array_partition variable=local_sol_y dim=2 cyclic factor=32

        int num_rows = (row_size+tile_size-1)/tile_size;
        int num_cols = (col_size+tile_size-1)/tile_size;

        for (unsigned i=0; i<num_rows*num_cols+2; i++) {
            int load_flag = i < num_rows*num_cols;
            int compute_flag = i > 0 && i < num_rows*num_cols+1;
            int store_flag = i > 1;
            size_t load_row = MIN(tile_size+2, row_size-ROW_IDX(i, num_cols)*tile_size);
            size_t load_col = MIN(tile_size+2, col_size-COL_IDX(i, num_cols)*tile_size);
            size_t compute_row = MIN(tile_size+2, row_size-ROW_IDX(i-1, num_cols)*tile_size);
            size_t compute_col = MIN(tile_size+2, col_size-COL_IDX(i-1, num_cols)*tile_size);
            size_t store_row = MIN(tile_size+2, row_size-ROW_IDX(i-2, num_cols)*tile_size);
            size_t store_col = MIN(tile_size+2, col_size-COL_IDX(i-2, num_cols)*tile_size);

            if (i % 2 == 0) {
                load(load_flag, local_orig_x, orig+ROW_IDX(i, num_cols)*col_size*tile_size+COL_IDX(i, num_cols)*tile_size, load_row, load_col);
                compute(compute_flag, local_orig_y, local_sol_y, local_filter, compute_row, compute_col);
                store(store_flag, sol+ROW_IDX(i-2, num_cols)*col_size*tile_size+COL_IDX(i-2, num_cols)*tile_size, local_sol_x, store_row, store_col);
            }
            else {
                load(load_flag, local_orig_y, orig+ROW_IDX(i, num_cols)*col_size*tile_size+COL_IDX(i, num_cols)*tile_size, load_row, load_col);
                compute(compute_flag, local_orig_x, local_sol_x, local_filter, compute_row, compute_col);
                store(store_flag, sol+ROW_IDX(i-2, num_cols)*col_size*tile_size+COL_IDX(i-2, num_cols)*tile_size, local_sol_y, store_row, store_col);
            }
        }

        return;
}
