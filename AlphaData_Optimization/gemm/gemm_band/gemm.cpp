#include "gemm.h"
#include <inttypes.h>

#include "ap_int.h" 

extern "C" {

#define TYPE_DOUBLE ap_int<512>
#define WIDTH_FACTOR_DOUBLE 8

void load(int flag, int i, int j, int k, TYPE_DOUBLE local_ma[T][T/WIDTH_FACTOR_DOUBLE], TYPE_DOUBLE local_mb[T][T/WIDTH_FACTOR_DOUBLE], TYPE_DOUBLE ma[N/WIDTH_FACTOR_DOUBLE], TYPE_DOUBLE mb[N/WIDTH_FACTOR_DOUBLE]){
#pragma HLS INLINE off
    int ii, jj, kk;
    if(flag){
        for(ii = 0; ii < tile_size; ii++) {
            for(kk = 0; kk < tile_size/WIDTH_FACTOR_DOUBLE; kk++) {
#pragma HLS PIPELINE II = 1
                local_ma[ii][kk] = ma[(i+ii)*col_size/WIDTH_FACTOR_DOUBLE + (k/WIDTH_FACTOR_DOUBLE+kk)];
	    }
	}

        for(kk = 0; kk < tile_size; kk++) {
            for(jj = 0; jj < tile_size/WIDTH_FACTOR_DOUBLE; jj++) {
#pragma HLS PIPELINE II = 1
                local_mb[kk][jj] = mb[(k+kk)*col_size/WIDTH_FACTOR_DOUBLE + (j/WIDTH_FACTOR_DOUBLE+jj)];
	    }
	}
   }
} 
void compute(int flag, TYPE_DOUBLE local_ma[T][T/WIDTH_FACTOR_DOUBLE], TYPE_DOUBLE local_mb[T][T/WIDTH_FACTOR_DOUBLE], TYPE_DOUBLE local_prod[T][T/WIDTH_FACTOR_DOUBLE]){
#pragma HLS INLINE off
    int ii, jj, kk, uu;
    if(flag) {
        for(kk = 0; kk < tile_size; kk++) {
            for(ii = 0; ii < tile_size; ii++) {
            #pragma HLS PIPELINE II = 1
                for(jj = 0; jj < tile_size; jj+=unroll_size) {
		    int range_idx = kk%WIDTH_FACTOR_DOUBLE*64;
		    uint64_t tmp = local_ma[ii][kk/WIDTH_FACTOR_DOUBLE].range(range_idx+63, range_idx);
		    TYPE mult1 = *((double *)(&tmp));
                    for(uu = 0; uu < unroll_size; uu++) {
		    #pragma HLS UNROLL
		    #pragma HLS DEPENDENCE variable="local_prod" inter false
			int prod_idx = (jj+uu)%WIDTH_FACTOR_DOUBLE*64;
			uint64_t tmp_mult2 = local_mb[kk][(jj+uu)/WIDTH_FACTOR_DOUBLE].range(prod_idx+63, prod_idx);
			TYPE mult2 = *((double *)(&tmp_mult2));
                        uint64_t tmp_prod = local_prod[ii][(jj+uu)/WIDTH_FACTOR_DOUBLE].range(prod_idx+63, prod_idx);
			TYPE prod = *((double *)(&tmp_prod));
			prod += mult1 * mult2;
			local_prod[ii][(jj+uu)/WIDTH_FACTOR_DOUBLE].range(prod_idx+63, prod_idx) = *((uint64_t *)(&prod));
                    }
		}
	    }
	}
    }
} 

void workload(TYPE_DOUBLE* ma, TYPE_DOUBLE* mb, TYPE_DOUBLE* prod){
#pragma HLS INTERFACE m_axi port=ma offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=mb offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=prod offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=ma bundle=control
#pragma HLS INTERFACE s_axilite port=mb bundle=control
#pragma HLS INTERFACE s_axilite port=prod bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int d,f;


    TYPE_DOUBLE local_ma_ping  [tile_size][tile_size/WIDTH_FACTOR_DOUBLE];
    TYPE_DOUBLE local_mb_ping  [tile_size][tile_size/WIDTH_FACTOR_DOUBLE];
#pragma HLS ARRAY_PARTITION variable=local_mb_ping complete dim=2
    TYPE_DOUBLE local_ma_pong  [tile_size][tile_size/WIDTH_FACTOR_DOUBLE];
    TYPE_DOUBLE local_mb_pong  [tile_size][tile_size/WIDTH_FACTOR_DOUBLE];
#pragma HLS ARRAY_PARTITION variable=local_mb_pong complete dim=2
    TYPE_DOUBLE local_prod[tile_size][tile_size/WIDTH_FACTOR_DOUBLE];
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
                for(jj = 0; jj < tile_size/WIDTH_FACTOR_DOUBLE; jj++) {
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
                for(jj = 0; jj < tile_size/WIDTH_FACTOR_DOUBLE; jj++) {
#pragma HLS PIPELINE II = 1
                    prod[(i+ii)*col_size/WIDTH_FACTOR_DOUBLE + (j/WIDTH_FACTOR_DOUBLE+jj)] = local_prod[ii][jj];
		}
	    }

        }
    return;
}

}
