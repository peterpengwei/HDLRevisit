#include "viterbi.h"
#include <string.h>

#include "ap_int.h"

typedef ap_int<128> uint128_t;
typedef ap_int<256> uint256_t;
typedef ap_int<512> uint512_t;

extern "C" {

#define WIDTH_FACTOR_FLOAT 4
#define WIDTH_FACTOR_CHAR 4

#define TYPE_FLOAT ap_int<128>
#define TYPE_CHAR ap_int<32>

#define JOBS_PER_BATCH 256
#define UNROLL_FACTOR 32
#define JOBS_PER_PE ((JOBS_PER_BATCH)/(UNROLL_FACTOR))

int viterbi( tok_t obs[N_OBS], prob_t init[N_STATES], prob_t transition[N_STATES*N_STATES], prob_t emission[N_STATES*N_TOKENS], state_t path[N_OBS] )
{
  prob_t llike[N_OBS][N_STATES];
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;
  // All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) )
 
  // Initialize with first observation and initial probabilities
  L_init: for( s=0; s<N_STATES; s++ ) {
  #pragma HLS PIPELINE
    llike[0][s] = init[s] + emission[s*N_TOKENS+obs[0]];
  }

  // Iteratively compute the probabilities over time
  L_timestep: for( t=1; t<N_OBS; t++ ) {
    L_curr_state: for( curr=0; curr<N_STATES; curr++ ) {
    #pragma HLS UNROLL
      // Compute likelihood HMM is in current state and where it came from.
      prob_t emit_prob = emission[curr*N_TOKENS+obs[t]];
      min_p = llike[t-1][0] + transition[curr] + emit_prob;
      L_prev_state: for( prev=1; prev<N_STATES; prev++ ) {
      #pragma HLS UNROLL
        p = llike[t-1][prev] +
            transition[prev*N_STATES+curr] +
            emit_prob;
	if (p < min_p) {
	  min_p = p;
	}
      }
      llike[t][curr] = min_p;
    }
  }

  // Identify end state
  min_s = 0;
  min_p = llike[N_OBS-1][min_s];
  L_end: for( s=1; s<N_STATES; s++ ) {
  #pragma HLS PIPELINE
    p = llike[N_OBS-1][s];
    if( p<min_p ) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  // Backtrack to recover full path
  L_backtrack: for( t=N_OBS-2; t>=0; t-- ) {
    min_p = llike[t][0] + transition[path[t+1]];
    min_s = 0;
    L_state: for( s=1; s<N_STATES; s++ ) {
    #pragma HLS UNROLL
      p = llike[t][s] + transition[s*N_STATES+path[t+1]];
      if (p < min_p) {
        min_p = p;
        min_s = s;
      }
    }
    path[t] = min_s;
  }

  return 0;
}

void viterbi_tiling( TYPE_CHAR* global_obs, TYPE_FLOAT* global_init, TYPE_FLOAT* global_transition, TYPE_FLOAT* global_emission, TYPE_CHAR* global_path ) {
    tok_t obs[N_OBS*JOBS_PER_PE];
    prob_t init[N_STATES*JOBS_PER_PE];
    prob_t transition[TRANS_SIZE*JOBS_PER_PE];
    prob_t emission[EMIT_SIZE*JOBS_PER_PE];
    state_t path[N_OBS*JOBS_PER_PE];

    TYPE_CHAR tmp_char;
    TYPE_FLOAT tmp_float;
    uint32_t int_to_float;

    int i,j;
    for (i=0; i<N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR; i++) {
    #pragma HLS PIPELINE
      tmp_char = global_obs[i];
      for (j=0; j<WIDTH_FACTOR_CHAR; j++) {
        obs[i*WIDTH_FACTOR_CHAR+j] = tmp_char.range(7, 0);
        tmp_char = tmp_char >> 8;
      }
    }
    for (i=0; i<N_STATES*JOBS_PER_PE/WIDTH_FACTOR_FLOAT; i++) {
    #pragma HLS PIPELINE
      tmp_float = global_init[i];
      for (j=0; j<WIDTH_FACTOR_FLOAT; j++) {
        int_to_float = tmp_float.range(31, 0);
        init[i*WIDTH_FACTOR_FLOAT+j] = *((float *)(&int_to_float));
        tmp_float = tmp_float >> 32;
      }
    }
    for (i=0; i<TRANS_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT; i++) {
    #pragma HLS PIPELINE
      tmp_float = global_transition[i];
      for (j=0; j<WIDTH_FACTOR_FLOAT; j++) {
        int_to_float = tmp_float.range(31, 0);
        transition[i*WIDTH_FACTOR_FLOAT+j] = *((float *)(&int_to_float));
        tmp_float = tmp_float >> 32;
      }
    }
    for (i=0; i<EMIT_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT; i++) {
    #pragma HLS PIPELINE
      tmp_float = global_emission[i];
      for (j=0; j<WIDTH_FACTOR_FLOAT; j++) {
        int_to_float = tmp_float.range(31, 0);
        emission[i*WIDTH_FACTOR_FLOAT+j] = *((float *)(&int_to_float));
        tmp_float = tmp_float >> 32;
      }
    }

    for (j=0; j<JOBS_PER_PE; j++) {
        viterbi(obs + j*N_OBS, init + j*N_STATES, transition + j*TRANS_SIZE, emission + j*EMIT_SIZE, path + j*N_OBS);
    }

    for (i=0; i<N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR; i++) {
    #pragma HLS PIPELINE
      tmp_char.range(WIDTH_FACTOR_CHAR*8-1, WIDTH_FACTOR_CHAR*8-8) = path[i*WIDTH_FACTOR_CHAR];
      for (j=0; j<WIDTH_FACTOR_CHAR; j++) {
        tmp_char = tmp_char >> 8;
      tmp_char.range(WIDTH_FACTOR_CHAR*8-1, WIDTH_FACTOR_CHAR*8-8) = path[i*WIDTH_FACTOR_CHAR+j];
      }
      global_path[i] = tmp_char;
    }
}

void buffer_load(int flag,  TYPE_CHAR* global_buf_A,  TYPE_CHAR part_buf_A[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR],
		           TYPE_FLOAT* global_buf_B, TYPE_FLOAT part_buf_B[UNROLL_FACTOR][N_STATES*JOBS_PER_PE/WIDTH_FACTOR_FLOAT],
		           TYPE_FLOAT* global_buf_C, TYPE_FLOAT part_buf_C[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT],
		           TYPE_FLOAT* global_buf_D, TYPE_FLOAT part_buf_D[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT]
		) {
#pragma HLS INLINE off
  int i;
  if (flag) {
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_A[i], global_buf_A + i * (N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR), sizeof(tok_t)*N_OBS*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_B[i], global_buf_B + i * (N_STATES*JOBS_PER_PE/WIDTH_FACTOR_FLOAT), sizeof(prob_t)*N_STATES*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_C[i], global_buf_C + i * (TRANS_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT), sizeof(prob_t)*TRANS_SIZE*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_D[i], global_buf_D + i * (EMIT_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT), sizeof(prob_t)*EMIT_SIZE*JOBS_PER_PE);
    }
  }
  return;
}

void buffer_store(int flag, TYPE_CHAR* global_buf_A, TYPE_CHAR part_buf_A[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR]) {
#pragma HLS INLINE off
  if (flag) {
    for (int i=0; i<UNROLL_FACTOR; i++) {
      memcpy(global_buf_A + i * (N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR), part_buf_A[i], sizeof(state_t)*N_OBS*JOBS_PER_PE);
    }
  }
  return;
}

void buffer_compute(int flag,  TYPE_CHAR local_obs[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR],
	                      TYPE_FLOAT local_init[UNROLL_FACTOR][N_STATES*JOBS_PER_PE/WIDTH_FACTOR_FLOAT],
		              TYPE_FLOAT local_transition[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT],      
                              TYPE_FLOAT local_emission[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT],
                               TYPE_CHAR local_path[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR]) {
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j=0; j<UNROLL_FACTOR; j++) {
    #pragma HLS UNROLL
	viterbi_tiling(local_obs[j], local_init[j], local_transition[j], local_emission[j], local_path[j]);
    }
  }
  return;
}

void workload( TYPE_CHAR* obs, TYPE_FLOAT* init, TYPE_FLOAT* transition, TYPE_FLOAT* emission, TYPE_CHAR* path, int num_jobs ) {
#pragma HLS INTERFACE m_axi port=obs offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=init offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=transition offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=emission offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=obs bundle=control
#pragma HLS INTERFACE s_axilite port=init bundle=control
#pragma HLS INTERFACE s_axilite port=transition bundle=control
#pragma HLS INTERFACE s_axilite port=emission bundle=control
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=num_jobs bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    TYPE_CHAR local_obs_x[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR];
    #pragma HLS ARRAY_PARTITION variable=local_obs_x complete dim=1
    TYPE_CHAR local_obs_y[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR];
    #pragma HLS ARRAY_PARTITION variable=local_obs_y complete dim=1

    TYPE_FLOAT local_init_x[UNROLL_FACTOR][N_STATES*JOBS_PER_PE/WIDTH_FACTOR_FLOAT];
    #pragma HLS ARRAY_PARTITION variable=local_init_x complete dim=1
    TYPE_FLOAT local_init_y[UNROLL_FACTOR][N_STATES*JOBS_PER_PE/WIDTH_FACTOR_FLOAT];
    #pragma HLS ARRAY_PARTITION variable=local_init_y complete dim=1

    TYPE_FLOAT local_transition_x[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT];
    #pragma HLS ARRAY_PARTITION variable=local_transition_x complete dim=1
    TYPE_FLOAT local_transition_y[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT];
    #pragma HLS ARRAY_PARTITION variable=local_transition_y complete dim=1

    TYPE_FLOAT local_emission_x[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT];
    #pragma HLS ARRAY_PARTITION variable=local_emission_x complete dim=1
    TYPE_FLOAT local_emission_y[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE/WIDTH_FACTOR_FLOAT];
    #pragma HLS ARRAY_PARTITION variable=local_emission_y complete dim=1

    TYPE_CHAR local_path_x[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR];
    #pragma HLS ARRAY_PARTITION variable=local_path_x complete dim=1
    TYPE_CHAR local_path_y[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/WIDTH_FACTOR_CHAR];
    #pragma HLS ARRAY_PARTITION variable=local_path_y complete dim=1


    int num_batches = num_jobs / JOBS_PER_BATCH;

    int i;
    for (i=0; i<num_batches+2; i++) {
      int load_flag = i >= 0 && i < num_batches;
      int compute_flag = i >= 1 && i < num_batches+1;
      int store_flag = i >= 2 && i < num_batches+2;
      if (i % 2 == 0) {
        buffer_load(load_flag, obs+i*JOBS_PER_BATCH*N_OBS/WIDTH_FACTOR_CHAR, local_obs_x, init+i*JOBS_PER_BATCH*N_STATES/WIDTH_FACTOR_FLOAT, local_init_x, transition+i*JOBS_PER_BATCH*TRANS_SIZE/WIDTH_FACTOR_FLOAT, local_transition_x, emission+i*JOBS_PER_BATCH*EMIT_SIZE/WIDTH_FACTOR_FLOAT, local_emission_x);
        buffer_compute(compute_flag, local_obs_y, local_init_y, local_transition_y, local_emission_y, local_path_y);
        buffer_store(store_flag, path+(i-2)*JOBS_PER_BATCH/WIDTH_FACTOR_CHAR, local_path_x);
      } 
      else {
        buffer_load(load_flag, obs+i*JOBS_PER_BATCH*N_OBS/WIDTH_FACTOR_CHAR, local_obs_y, init+i*JOBS_PER_BATCH*N_STATES/WIDTH_FACTOR_FLOAT, local_init_y, transition+i*JOBS_PER_BATCH*TRANS_SIZE/WIDTH_FACTOR_FLOAT, local_transition_y, emission+i*JOBS_PER_BATCH*EMIT_SIZE/WIDTH_FACTOR_FLOAT, local_emission_y);
        buffer_compute(compute_flag, local_obs_x, local_init_x, local_transition_x, local_emission_x, local_path_x);
        buffer_store(store_flag, path+(i-2)*JOBS_PER_BATCH/WIDTH_FACTOR_CHAR, local_path_y);
      } 
    }
    return;
}

}
