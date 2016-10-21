#include "gemm.h"

void gemm( TYPE m1[N], TYPE m2[N], TYPE prod[N] ){
    int i, j, k;
    int k_col, i_col;
    TYPE mult;

outer:for(i=0;i<row_size;i++) {
middle:for(j=0;j<col_size;j++) {
           i_col = i * col_size;
           TYPE sum = 0;
inner:for(k=0;k<row_size;k++) {
          k_col = k * col_size;
          mult = m1[i_col + k] * m2[k_col + j];
          sum += mult;
      }
      prod[i_col + j]  = sum;
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



    TYPE local_m1  [tile_size][tile_size];
    TYPE local_m2  [tile_size][tile_size];
#pragma HLS ARRAY_PARTITION variable=local_m2 complete dim=2
    TYPE local_prod[tile_size][tile_size];
#pragma HLS ARRAY_PARTITION variable=local_prod complete dim=2
    //array partition

    TYPE mult, sum;
    int i, j , k, ii, jj, kk;
    int uu;
    for(i = 0; i < row_size; i += tile_size)
        for(j = 0; j < col_size; j += tile_size){

            
            for(ii = 0; ii < tile_size; ii++)
                for(jj = 0; jj < tile_size; jj++)
#pragma HLS PIPELINE
                    local_prod[ii][jj] = 0;


            for(k = 0; k < col_size; k += tile_size){

                for(ii = 0; ii < tile_size; ii++)
                    for(kk = 0; kk < tile_size; kk++)
#pragma HLS PIPELINE
                        local_m1[ii][kk] = m1[(i+ii)*col_size + (k+kk)];

                for(kk = 0; kk < tile_size; kk++)
                    for(jj = 0; jj < tile_size; jj++)
#pragma HLS PIPELINE
                        local_m2[kk][jj] = m2[(k+kk)*col_size + (j+jj)];

                for(kk = 0; kk < tile_size; kk++)
                    for(ii = 0; ii < tile_size; ii++)
                        for(jj = 0; jj < tile_size; jj+= unroll_size)
#pragma HLS PIPELINE II = 20
                            for(uu = 0; uu < unroll_size; uu++)
#pragma HLS UNROLL
#pragma HLS DEPENDENCE variable="local_prod" inter false
                            local_prod[ii][jj+uu] += local_m1[ii][kk] * local_m2[kk][jj+uu];
            }
            for(ii = 0; ii < tile_size; ii++)
                for(jj = 0; jj < tile_size; jj++)
#pragma HLS PIPELINE
                    prod[(i+ii)*col_size + (j+jj)] = local_prod[ii][jj];
            
        }
    //write back c block
    //gemm(m1, m2, prod);
    return;
}
