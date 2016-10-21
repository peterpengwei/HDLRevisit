/*
Implementation based on http://www-igm.univ-mlv.fr/~lecroq/string/node8.html
*/

/*
void CPF(char pattern[PATTERN_SIZE], int32_t kmpNext[PATTERN_SIZE]) {
    int32_t k, q;
    k = 0;
    kmpNext[0] = 0;

    c1 : for(q = 1; q < PATTERN_SIZE; q++){
        c2 : while(k > 0 && pattern[k] != pattern[q]){
            k = kmpNext[q];
        }
        if(pattern[k] == pattern[q]){
            k++;
        }
        kmpNext[q] = k;
    }
}

int kmp(char pattern[PATTERN_SIZE], char input[STRING_SIZE], int32_t kmpNext[PATTERN_SIZE], int32_t n_matches[1]) {
    int32_t i, q;
    n_matches[0] = 0;

    CPF(pattern, kmpNext);

    q = 0;
    k1 : for(i = 0; i < STRING_SIZE; i++){
        k2 : while (q > 0 && pattern[q] != input[i]){
            q = kmpNext[q];
        }
        if (pattern[q] == input[i]){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmpNext[q - 1];
        }
    }
    return 0;
}
*/
#include "kmp.h"
#include <string.h>

extern "C" {

void kmp(char pattern[PATTERN_SIZE], char input[CACHE_SIZE/FACT], int& n_matches) {
#pragma HLS inline off

    char input_local[PATTERN_SIZE];
#pragma HLS ARRAY_PARTITION variable=input_local complete dim=1
    char pattern_local[PATTERN_SIZE];
#pragma HLS ARRAY_PARTITION variable=pattern_local complete dim=1
//     bool is_match[PATTERN_SIZE];
// #pragma HLS ARRAY_PARTITION variable=is_match complete dim=1

    int i, j;
    for (i=0; i<PATTERN_SIZE; i++) {
#pragma HLS UNROLL
	pattern_local[i] = pattern[i];
	input_local[i] = input[i];
    }
    bool is_match;
    for(i=0; i<CACHE_SIZE/FACT-PATTERN_SIZE+1; i++){
#pragma HLS PIPELINE
	is_match = true;
        for(j=0; j<PATTERN_SIZE; j++){
#pragma HLS UNROLL
	  is_match = is_match && (pattern_local[j] == input_local[j]);
        }
        if( is_match )
            n_matches = n_matches + 1;
	for (j=0; j<PATTERN_SIZE-1; j++) {
	  input_local[j] = input_local[j+1];
	}
	input_local[PATTERN_SIZE-1] = input[i+PATTERN_SIZE];
    }
}

void buffer_load(bool flag, char local_buf[FACT][CACHE_SIZE/FACT], char* global_buf) {
#pragma HLS inline off
  int j;
  if (flag) {
      for(j=0; j<FACT; j++){
          memcpy((void*)local_buf[j], (const void*)(global_buf+j*CACHE_SIZE/FACT), sizeof(char)*(CACHE_SIZE/FACT));
      }
  }
}

void buffer_compute(bool flag, char local_buf[FACT][CACHE_SIZE/FACT], char pattern_buf[FACT][PATTERN_SIZE], int n_matches_buf[FACT]) {
#pragma HLS inline off
  int j;
  if (flag) {
      for(j=0; j<FACT; j++){
#pragma HLS UNROLL
          kmp(pattern_buf[j], local_buf[j], n_matches_buf[j]);
      }
  }

}

void workload(char pattern[PATTERN_SIZE], char input[STRING_SIZE], int32_t n_matches[1]) {
#pragma HLS INTERFACE m_axi port=pattern offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=n_matches offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=pattern bundle=control
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=n_matches bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    char pattern_buf[FACT][PATTERN_SIZE];
#pragma HLS ARRAY_PARTITION variable=pattern_buf complete dim=1
    char input_buf_x[FACT][CACHE_SIZE/FACT];
#pragma HLS ARRAY_PARTITION variable=input_buf_x complete dim=1
    char input_buf_y[FACT][CACHE_SIZE/FACT];
#pragma HLS ARRAY_PARTITION variable=input_buf_y complete dim=1
    int32_t n_matches_buf[FACT];
#pragma HLS ARRAY_PARTITION variable=n_matches_buf complete dim=1

    int i, j;
    for (i=0; i<FACT; i++) {
#pragma HLS UNROLL
	n_matches_buf[i] = 0;
    }
    memcpy(pattern_buf[0], pattern, sizeof(char)*(PATTERN_SIZE));
    for (j=0; j<PATTERN_SIZE; j++) {
#pragma HLS PIPELINE
        for (i=1; i<FACT; i++) {
#pragma HLS UNROLL
	    pattern_buf[i][j] = pattern_buf[0][j];
        }
    }

    for(i=0; i<STRING_SIZE/CACHE_SIZE+1; i++){
	if (i % 2 == 0) {
	    buffer_load(i < STRING_SIZE/CACHE_SIZE, input_buf_x, input + i*CACHE_SIZE);
	    buffer_compute(i > 0, input_buf_y, pattern_buf, n_matches_buf);
	}
	else {
	    buffer_load(i < STRING_SIZE/CACHE_SIZE, input_buf_y, input + i*CACHE_SIZE);
	    buffer_compute(i > 0, input_buf_x, pattern_buf, n_matches_buf);
	}
    }

    int final_res = 0;
    for (i=0; i<FACT; i++) {
#pragma HLS UNROLL
	final_res += n_matches_buf[i];
    }
    n_matches[0] = final_res;

    return;
}

}
