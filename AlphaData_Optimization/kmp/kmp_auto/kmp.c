#include "kmp.h"
#include <assert.h>

#define UNROLL_FACTOR 8
#define PE_SIZE ((CACHE_SIZE+UNROLL_FACTOR-1)/UNROLL_FACTOR)
#define CYCLIC_FACTOR 4
#define BLOCK_SIZE ((PE_SIZE+PATTERN_SIZE-1+CYCLIC_FACTOR-1)/CYCLIC_FACTOR)

int kmp(char* pattern, char input[][CYCLIC_FACTOR], size_t size) {
#pragma HLS inline off
    int str_cnt = 0;
    char data[PATTERN_SIZE+CYCLIC_FACTOR-1];
#pragma HLS array_partition variable=data complete
    char flags[CYCLIC_FACTOR];
#pragma HLS array_partition variable=flags complete
    int i, j, k;
    for(i=0; i<PE_SIZE+PATTERN_SIZE-1; i+=CYCLIC_FACTOR){
#pragma HLS pipeline
	for (j=0; j<CYCLIC_FACTOR; j++) {
#pragma HLS unroll
	    if (i+j < size) {
	        data[PATTERN_SIZE-1+j] = input[(i+j)/CYCLIC_FACTOR][j];
	    }
	}
	for (j=0; j<CYCLIC_FACTOR; j++) {
#pragma HLS unroll
	    if (i-(PATTERN_SIZE-1)+j >= 0 && i-(PATTERN_SIZE-1)+j <= size-PATTERN_SIZE) {
	        int char_cnt = 0;
		for (k=0; k<PATTERN_SIZE; k++) {
#pragma HLS unroll
		    if (data[j+k] == pattern[k]) char_cnt++;
		}
		flags[j] = char_cnt == PATTERN_SIZE? 1:0;
	    }
	    else flags[j] = 0;
	}
	for (j=0; j<CYCLIC_FACTOR; j++) {
#pragma HLS unroll
	    str_cnt += flags[j];
	}
	for (j=0; j<PATTERN_SIZE-1; j++) {
#pragma HLS unroll
	    data[j] = data[CYCLIC_FACTOR+j];
	}
    }
    return str_cnt;
}

void load(int flag, uint8_t dst[][BLOCK_SIZE][CYCLIC_FACTOR], uint8_t* src, size_t size) {
#pragma HLS inline off
    int i, j;
    if (flag) {
        for (i=0; i<UNROLL_FACTOR; i++) {
	  int pe_size = (int) size - i*PE_SIZE;
	  if (pe_size > PE_SIZE+PATTERN_SIZE-1) pe_size = PE_SIZE+PATTERN_SIZE-1;
	  if (pe_size > 0) {
	    for (j=0; j<pe_size; j++) {
#pragma HLS pipeline
		dst[i][j/CYCLIC_FACTOR][j%CYCLIC_FACTOR] = src[i*PE_SIZE+j];
	    }
	  }
        }
    }
}

int compute(int flag, char* pattern, char input[][BLOCK_SIZE][CYCLIC_FACTOR], size_t size) {
#pragma HLS inline off
  int sum = 0;
  if (flag) {
    char patterns[UNROLL_FACTOR][PATTERN_SIZE];
#pragma HLS array_partition variable=patterns dim=1 complete
#pragma HLS array_partition variable=patterns dim=2 complete
    int i,j;
    for (i=0; i<UNROLL_FACTOR; i++) {
#pragma HLS unroll
      for (j=0; j<PATTERN_SIZE; j++) {
#pragma HLS unroll
        patterns[i][j] = pattern[j];
      }
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
#pragma HLS unroll
        int pe_size = (int) size - i*PE_SIZE;
        if (pe_size > PE_SIZE+PATTERN_SIZE-1) pe_size = PE_SIZE+PATTERN_SIZE-1;
        sum += pe_size >= PATTERN_SIZE? kmp(patterns[i], input[i], pe_size) : 0;
    }
  }
  return sum;
}

void workload(char* pattern, char* input, int32_t n_matches[1]) {
#pragma HLS INTERFACE m_axi port=pattern offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=n_matches offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=pattern bundle=control
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=n_matches bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    char pattern_buf[PATTERN_SIZE];
#pragma HLS array_partition variable=pattern_buf complete
    memcpy(pattern_buf, pattern, sizeof(char)*(PATTERN_SIZE));

    int num_batches = (STRING_SIZE+CACHE_SIZE-1)/CACHE_SIZE;
    int tail_size = STRING_SIZE % CACHE_SIZE;
    if (tail_size == 0) tail_size = CACHE_SIZE;

    char input_buf_x[UNROLL_FACTOR][BLOCK_SIZE][CYCLIC_FACTOR];
#pragma HLS array_partition variable=input_buf_x dim=1 complete
#pragma HLS array_partition variable=input_buf_x dim=3 complete
    char input_buf_y[UNROLL_FACTOR][BLOCK_SIZE][CYCLIC_FACTOR];
#pragma HLS array_partition variable=input_buf_y dim=1 complete
#pragma HLS array_partition variable=input_buf_y dim=3 complete

    int result = 0;

    int i;
    for(i=0; i<num_batches+1; i++){
        int load_size = i == num_batches-1? tail_size:CACHE_SIZE+PATTERN_SIZE-1;
        int compute_size = i == num_batches? tail_size:CACHE_SIZE+PATTERN_SIZE-1;
        int load_flag = i < num_batches;
        int compute_flag = i > 0;
        if (i % 2 == 0) {
          load(load_flag, input_buf_x, input+i*CACHE_SIZE, load_size);
          result += compute(compute_flag, pattern_buf, input_buf_y, compute_size);
        }
        else {
          load(load_flag, input_buf_y, input+i*CACHE_SIZE, load_size);
          result += compute(compute_flag, pattern_buf, input_buf_x, compute_size);
        }
    }
    n_matches[0] = result;
    return;
}
