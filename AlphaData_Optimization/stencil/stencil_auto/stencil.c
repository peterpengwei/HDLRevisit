#include "stencil.h"
#include <assert.h>
#include <string.h>

#define UNROLL_R 3
// UNROLL_R is a constant that is NOT for DSE!!!
#define UNROLL_C 1
#define UNROLL_IMG 2
// UNROLL_C and UNROLL_IMG are the fine-grained and coarse-grained unroll factor to explore

#define MIN(A,B) ((A)<(B)?(A):(B))
#define ROW_IDX(A, num_cols) ((A)/(num_cols))
#define COL_IDX(A, num_cols) ((A)%(num_cols))

void stencil (TYPE orig[][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C], TYPE sol[tile_size][(tile_size+UNROLL_C-1)/UNROLL_C][UNROLL_C], TYPE filter[]){
#pragma HLS inline off

    int r, c, i, j;
    TYPE data[UNROLL_R][UNROLL_C+2];
#pragma HLS array_partition variable=data dim=1 complete
#pragma HLS array_partition variable=data dim=2 complete

    stencil_label1:for (r=0; r<tile_size; r++) {
    stencil_label2:for (c=0; c<tile_size+2; c+=(UNROLL_C)) {
#pragma HLS pipeline
	for (j=0; j<UNROLL_C; j++) {
#pragma HLS unroll
	    if (c+j < tile_size+2) {
		switch (r % 3) {
		case 0:
		    data[0][j+2] = orig[0][j][(r+0)/UNROLL_R][(c+j)/UNROLL_C];
		    data[1][j+2] = orig[1][j][(r+1)/UNROLL_R][(c+j)/UNROLL_C];
		    data[2][j+2] = orig[2][j][(r+2)/UNROLL_R][(c+j)/UNROLL_C];
		    break;
		case 1:
		    data[0][j+2] = orig[1][j][(r+0)/UNROLL_R][(c+j)/UNROLL_C];
		    data[1][j+2] = orig[2][j][(r+1)/UNROLL_R][(c+j)/UNROLL_C];
		    data[2][j+2] = orig[0][j][(r+2)/UNROLL_R][(c+j)/UNROLL_C];
		    break;
		default:
		    data[0][j+2] = orig[2][j][(r+0)/UNROLL_R][(c+j)/UNROLL_C];
		    data[1][j+2] = orig[0][j][(r+1)/UNROLL_R][(c+j)/UNROLL_C];
		    data[2][j+2] = orig[1][j][(r+2)/UNROLL_R][(c+j)/UNROLL_C];
		}
	    }
	}
	for (j=0; j<UNROLL_C; j++) {
#pragma HLS unroll
	  if (c-2+j < tile_size && c-2+j >= 0) {
             sol[r][(c-2+j)/UNROLL_C][UNROLL_C==1?0:(j>=2?j-2:UNROLL_C+j-2)] = filter[0] * data[0][j+0]
	     	                                             + filter[1] * data[0][j+1]
	     	                                             + filter[2] * data[0][j+2]
	     	                                             + filter[3] * data[1][j+0]
	     	                                             + filter[4] * data[1][j+1]
	     	                                             + filter[5] * data[1][j+2]
	     	                                             + filter[6] * data[2][j+0]
	     	                                             + filter[7] * data[2][j+1]
	     	                                             + filter[8] * data[2][j+2];
	  }
	}
	for (i=0; i<UNROLL_R; i++) {
#pragma HLS unroll
	for (j=0; j<2; j++) {
#pragma HLS unroll
	  data[i][j] = data[i][j+UNROLL_C];
	}
	}
    }
    }
}

void load(int flag, TYPE dst[][UNROLL_R][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C], TYPE* src, int num_imgs) {
#pragma HLS inline off
    if (flag) {
	assert (num_imgs <= UNROLL_IMG);
	for (int k=0; k<num_imgs; k++) {
            for (int i=0; i<tile_size+2; i++) {
	    for (int j=0; j<tile_size+2; j++) {
#pragma HLS pipeline
	        dst[k][i%UNROLL_R][j%UNROLL_C][i/UNROLL_R][j/UNROLL_C] = src[k*(tile_size+2)*(tile_size+2)+i*(tile_size+2)+j];
	    }
            }
	}
    }
}

void store(int flag, TYPE* dst, TYPE src[][tile_size][(tile_size+UNROLL_C-1)/UNROLL_C][UNROLL_C], int num_imgs) {
#pragma HLS inline off
    if (flag) {
	for (int k=0; k<num_imgs; k++) {
            for (int i=0; i<tile_size; i++) {
	    for (int j=0; j<tile_size; j++) {
#pragma HLS pipeline
	        dst[k*tile_size*tile_size+i*tile_size+j] = src[k][i][j/UNROLL_C][j%UNROLL_C];
	    }
            }
	}
    }
}

void compute(int flag, TYPE orig[][UNROLL_R][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C], TYPE sol[][tile_size][(tile_size+UNROLL_C-1)/UNROLL_C][UNROLL_C], TYPE filter[f_size], int num_imgs) {
#pragma HLS inline off
    if (flag) {
	assert (num_imgs <= UNROLL_IMG);
	TYPE filters[UNROLL_IMG][f_size];
#pragma HLS array_partition variable=filters dim=1 complete
#pragma HLS array_partition variable=filters dim=2 complete
	for (int i=0; i<UNROLL_IMG; i++) {
#pragma HLS unroll
	    for (int j=0; j<f_size; j++) {
#pragma HLS unroll
	        filters[i][j] = filter[j];
	    }
	}
	for (int k=0; k<UNROLL_IMG; k++) {
#pragma HLS unroll
	    if (k < num_imgs) {
	        stencil(orig[k], sol[k], filters[k]);
	    }
	}
    }

}

void workload(TYPE* orig, TYPE* sol, TYPE filter[f_size], int num_imgs){
#pragma HLS INTERFACE m_axi port=orig offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=sol offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=filter offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=orig bundle=control
#pragma HLS INTERFACE s_axilite port=sol bundle=control
#pragma HLS INTERFACE s_axilite port=filter bundle=control
#pragma HLS INTERFACE s_axilite port=num_imgs bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	assert (num_imgs == (1 << 12));
        TYPE local_filter[f_size];
#pragma HLS array_partition variable=local_filter complete
        memcpy(local_filter, filter, sizeof(TYPE)*f_size);

        TYPE local_orig_x[UNROLL_IMG][UNROLL_R][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C];
#pragma HLS array_partition variable=local_orig_x dim=1 complete
#pragma HLS array_partition variable=local_orig_x dim=2 complete
#pragma HLS array_partition variable=local_orig_x dim=3 complete
        TYPE local_sol_x[UNROLL_IMG][tile_size][(tile_size+UNROLL_C-1)/UNROLL_C][UNROLL_C];
#pragma HLS array_partition variable=local_sol_x dim=1 complete
#pragma HLS array_partition variable=local_sol_x dim=4 complete
        TYPE local_orig_y[UNROLL_IMG][UNROLL_R][UNROLL_C][(tile_size+2+UNROLL_R-1)/UNROLL_R][(tile_size+2+UNROLL_C-1)/UNROLL_C];
#pragma HLS array_partition variable=local_orig_y dim=1 complete
#pragma HLS array_partition variable=local_orig_y dim=2 complete
#pragma HLS array_partition variable=local_orig_y dim=3 complete
        TYPE local_sol_y[UNROLL_IMG][tile_size][(tile_size+UNROLL_C-1)/UNROLL_C][UNROLL_C];
#pragma HLS array_partition variable=local_sol_y dim=1 complete
#pragma HLS array_partition variable=local_sol_y dim=4 complete

	int num_batches = (num_imgs+UNROLL_IMG-1)/UNROLL_IMG;
	int tail_imgs = num_imgs % UNROLL_IMG;
	if (tail_imgs == 0) tail_imgs = UNROLL_IMG;

        for (unsigned i=0; i<num_batches+2; i++) {
            int load_flag = i < num_batches;
            int compute_flag = i > 0 && i < num_batches+1;
            int store_flag = i > 1;
	    int load_imgs = i == num_batches-1? tail_imgs:UNROLL_IMG;
	    int compute_imgs = i == num_batches? tail_imgs:UNROLL_IMG;
	    int store_imgs = i == num_batches+1? tail_imgs:UNROLL_IMG;
            if (i % 2 == 0) {
                load(load_flag, local_orig_x, orig+i*UNROLL_IMG*(tile_size+2)*(tile_size+2), load_imgs);
                compute(compute_flag, local_orig_y, local_sol_y, local_filter, compute_imgs);
                store(store_flag, sol+(i-2)*UNROLL_IMG*tile_size*tile_size, local_sol_x, store_imgs);
            }
            else {
                load(load_flag, local_orig_y, orig+i*UNROLL_IMG*(tile_size+2)*(tile_size+2), load_imgs);
                compute(compute_flag, local_orig_x, local_sol_x, local_filter, compute_imgs);
                store(store_flag, sol+(i-2)*UNROLL_IMG*tile_size*tile_size, local_sol_y, store_imgs);
            }
	}

        return;
}
