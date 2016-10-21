#include "gemm.h"
void load(int flag, int i, int j, int k, TYPE local_ma[T][T], TYPE local_mb[T][T], TYPE ma[N], TYPE mb[N]){
#pragma HLS INLINE off
    int ii, jj, kk;
    if(flag){
        for(ii = 0; ii < tile_size; ii++) {
            for(kk = 0; kk < tile_size; kk++) {
#pragma HLS PIPELINE II = 1
                local_ma[ii][kk] = ma[(i+ii)*col_size + (k+kk)];
	    }
	}

        for(kk = 0; kk < tile_size; kk++) {
            for(jj = 0; jj < tile_size; jj++) {
#pragma HLS PIPELINE II = 1
                local_mb[kk][jj] = mb[(k+kk)*col_size + (j+jj)];
	    }
	}
   }
} 
void compute(int flag, TYPE local_ma[T][T], TYPE local_mb[T][T], TYPE local_prod[T][T]){
#pragma HLS INLINE off
    int ii, jj, kk, uu;
    if(flag) {
        for(kk = 0; kk < tile_size; kk++) {
            for(ii = 0; ii < tile_size; ii++) {
                for(jj = 0; jj < tile_size; jj+=unroll_size) {
		#pragma HLS PIPELINE II = 1
                    for(uu = 0; uu < unroll_size; uu++) {
		    #pragma HLS UNROLL
		    #pragma HLS DEPENDENCE variable="local_prod" inter false
                        local_prod[ii][jj+uu] += local_ma[ii][kk] * local_mb[kk][jj+uu];
                    }
		}
	    }
	}
    }
} 

void workload(TYPE ma[N], TYPE mb[N], TYPE prod[N]){
#pragma HLS INTERFACE m_axi port=ma offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=mb offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=prod offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=ma bundle=control
#pragma HLS INTERFACE s_axilite port=mb bundle=control
#pragma HLS INTERFACE s_axilite port=prod bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int d,f;


    TYPE local_ma_ping  [tile_size][tile_size];
    TYPE local_mb_ping  [tile_size][tile_size];
#pragma HLS ARRAY_PARTITION variable=local_mb_ping complete dim=2
    TYPE local_ma_pong  [tile_size][tile_size];
    TYPE local_mb_pong  [tile_size][tile_size];
#pragma HLS ARRAY_PARTITION variable=local_mb_pong complete dim=2
    TYPE local_prod[tile_size][tile_size];
#pragma HLS ARRAY_PARTITION variable=local_prod complete dim=2


    //array partition

    TYPE mult, sum;
    int i, j , k, ii, jj, kk;
    int tile_num = col_size / tile_size;
    int index, load_flag, compute_flag;
    int m, n;
    int counter = 0;
    for(i = 0; i < row_size; i += tile_size)
        for(j = 0; j < col_size; j += tile_size){


            for(ii = 0; ii < tile_size; ii++) {
                for(jj = 0; jj < tile_size; jj++) {
#pragma HLS PIPELINE II = 1
                    local_prod[ii][jj] = 0;
		}
	    }


            for(index = 0; index < tile_num + 1; index++){

                if(counter == 0)
                {
                    load(index < tile_num, i, j, index * tile_size, local_ma_ping, local_mb_ping, ma, mb);
                    compute(index > 0, local_ma_pong, local_mb_pong, local_prod);
                }
                else{
                    load(index < tile_num, i, j, index * tile_size, local_ma_pong, local_mb_pong, ma, mb);
                    compute(index > 0, local_ma_ping, local_mb_ping, local_prod);
                }
                counter = counter + 1;
                if(counter == 2)
                    counter = 0;
            }
            for(ii = 0; ii < tile_size; ii++) {
                for(jj = 0; jj < tile_size; jj++) {
#pragma HLS PIPELINE II = 1
                    prod[(i+ii)*col_size + (j+jj)] = local_prod[ii][jj];
		}
	    }

        }
    return;
}
