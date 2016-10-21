#include "viterbi.h"
#include "ap_int.h"
#include <string.h>

typedef ap_int<256> uint256_t;

#define JOBS_PER_BATCH 1024
#define UNROLL_FACTOR 64
#define JOBS_PER_PE ((JOBS_PER_BATCH)/(UNROLL_FACTOR))

extern "C" {

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
      // Compute likelihood HMM is in current state and where it came from.
      L_prev_state: for( prev=0; prev<N_STATES; prev++ ) {
      #pragma HLS PIPELINE
        p = llike[t-1][prev] +
            transition[prev*N_STATES+curr] +
            emission[curr*N_TOKENS+obs[t]];
	if (!prev || p < min_p) {
	  min_p = p;
	}
        llike[t][curr] = min_p;
      }
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
    L_state: for( s=0; s<N_STATES; s++ ) {
    #pragma HLS PIPELINE II=2
      p = llike[t][s] + transition[s*N_STATES+path[t+1]];
      if (!s || p < min_p) {
        min_p = p;
        min_s = s;
      }
    }
    path[t] = min_s;
  }

  return 0;
}

void viterbi_tiling( tok_t* obs, prob_t* init, prob_t* transition, prob_t* emission, state_t* path ) {
#pragma HLS INLINE off
    for (int j=0; j<JOBS_PER_PE; j++) {
        viterbi(obs + j*N_OBS, init + j*N_STATES, transition + j*TRANS_SIZE, emission + j*EMIT_SIZE, path + j*N_OBS);
    }
}

void copy_A(uint256_t* large_A, tok_t* part_buf_A) {
  int i, j;
  for (i=0; i<JOBS_PER_PE; i++) {
    for (j=0; j<4; j++) {
    #pragma HLS PIPELINE
      uint256_t curr = large_A[i*4+j];
      part_buf_A[i*128+j*32+ 0] = (uint8_t)(curr >>   0);
      part_buf_A[i*128+j*32+ 1] = (uint8_t)(curr >>   8);
      part_buf_A[i*128+j*32+ 2] = (uint8_t)(curr >>  16);
      part_buf_A[i*128+j*32+ 3] = (uint8_t)(curr >>  24);
      part_buf_A[i*128+j*32+ 4] = (uint8_t)(curr >>  32);
      part_buf_A[i*128+j*32+ 5] = (uint8_t)(curr >>  40);
      part_buf_A[i*128+j*32+ 6] = (uint8_t)(curr >>  48);
      part_buf_A[i*128+j*32+ 7] = (uint8_t)(curr >>  56);
      part_buf_A[i*128+j*32+ 8] = (uint8_t)(curr >>  64);
      part_buf_A[i*128+j*32+ 9] = (uint8_t)(curr >>  72);
      part_buf_A[i*128+j*32+10] = (uint8_t)(curr >>  80);
      part_buf_A[i*128+j*32+11] = (uint8_t)(curr >>  88);
      part_buf_A[i*128+j*32+12] = (uint8_t)(curr >>  96);
      part_buf_A[i*128+j*32+13] = (uint8_t)(curr >> 104);
      part_buf_A[i*128+j*32+14] = (uint8_t)(curr >> 112);
      part_buf_A[i*128+j*32+15] = (uint8_t)(curr >> 120);
      part_buf_A[i*128+j*32+16] = (uint8_t)(curr >> 128);
      part_buf_A[i*128+j*32+17] = (uint8_t)(curr >> 136);
      part_buf_A[i*128+j*32+18] = (uint8_t)(curr >> 144);
      part_buf_A[i*128+j*32+19] = (uint8_t)(curr >> 152);
      part_buf_A[i*128+j*32+20] = (uint8_t)(curr >> 160);
      part_buf_A[i*128+j*32+21] = (uint8_t)(curr >> 168);
      part_buf_A[i*128+j*32+22] = (uint8_t)(curr >> 176);
      part_buf_A[i*128+j*32+23] = (uint8_t)(curr >> 184);
      part_buf_A[i*128+j*32+24] = (uint8_t)(curr >> 192);
      part_buf_A[i*128+j*32+25] = (uint8_t)(curr >> 200);
      part_buf_A[i*128+j*32+26] = (uint8_t)(curr >> 208);
      part_buf_A[i*128+j*32+27] = (uint8_t)(curr >> 216);
      part_buf_A[i*128+j*32+28] = (uint8_t)(curr >> 224);
      part_buf_A[i*128+j*32+29] = (uint8_t)(curr >> 232);
      part_buf_A[i*128+j*32+30] = (uint8_t)(curr >> 240);
      part_buf_A[i*128+j*32+31] = (uint8_t)(curr >> 248);
    }
  }
}

void copy_E(uint256_t* large_E, tok_t* part_buf_E) {
  int i, j;
  for (i=0; i<JOBS_PER_PE; i++) {
    for (j=0; j<4; j++) {
    #pragma HLS PIPELINE
      uint256_t curr = part_buf_E[i*128+j*32+31];
      curr = (curr << 8) | part_buf_E[i*128+j*32+30];
      curr = (curr << 8) | part_buf_E[i*128+j*32+29];
      curr = (curr << 8) | part_buf_E[i*128+j*32+28];
      curr = (curr << 8) | part_buf_E[i*128+j*32+27];
      curr = (curr << 8) | part_buf_E[i*128+j*32+26];
      curr = (curr << 8) | part_buf_E[i*128+j*32+25];
      curr = (curr << 8) | part_buf_E[i*128+j*32+24];
      curr = (curr << 8) | part_buf_E[i*128+j*32+23];
      curr = (curr << 8) | part_buf_E[i*128+j*32+22];
      curr = (curr << 8) | part_buf_E[i*128+j*32+21];
      curr = (curr << 8) | part_buf_E[i*128+j*32+20];
      curr = (curr << 8) | part_buf_E[i*128+j*32+19];
      curr = (curr << 8) | part_buf_E[i*128+j*32+18];
      curr = (curr << 8) | part_buf_E[i*128+j*32+17];
      curr = (curr << 8) | part_buf_E[i*128+j*32+16];
      curr = (curr << 8) | part_buf_E[i*128+j*32+15];
      curr = (curr << 8) | part_buf_E[i*128+j*32+14];
      curr = (curr << 8) | part_buf_E[i*128+j*32+13];
      curr = (curr << 8) | part_buf_E[i*128+j*32+12];
      curr = (curr << 8) | part_buf_E[i*128+j*32+11];
      curr = (curr << 8) | part_buf_E[i*128+j*32+10];
      curr = (curr << 8) | part_buf_E[i*128+j*32+9];
      curr = (curr << 8) | part_buf_E[i*128+j*32+8];
      curr = (curr << 8) | part_buf_E[i*128+j*32+7];
      curr = (curr << 8) | part_buf_E[i*128+j*32+6];
      curr = (curr << 8) | part_buf_E[i*128+j*32+5];
      curr = (curr << 8) | part_buf_E[i*128+j*32+4];
      curr = (curr << 8) | part_buf_E[i*128+j*32+3];
      curr = (curr << 8) | part_buf_E[i*128+j*32+2];
      curr = (curr << 8) | part_buf_E[i*128+j*32+1];
      curr = (curr << 8) | part_buf_E[i*128+j*32+0];
      large_E[i*4+j] = curr;
    }
  }
}

void buffer_load(int flag, uint256_t* global_buf_A,  tok_t part_buf_A[UNROLL_FACTOR][N_OBS*JOBS_PER_PE],
		             prob_t* global_buf_B, prob_t part_buf_B[UNROLL_FACTOR][N_STATES*JOBS_PER_PE],
		             prob_t* global_buf_C, prob_t part_buf_C[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE],
		             prob_t* global_buf_D, prob_t part_buf_D[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE]
		) {
#pragma HLS INLINE off
  uint256_t large_A[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/32];
  #pragma HLS ARRAY_PARTITION variable=large_A dim=1 complete
  int i;
  if (flag) {
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(large_A[i], global_buf_A + i * (N_OBS*JOBS_PER_PE)/32, sizeof(tok_t)*N_OBS*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_B[i], global_buf_B + i * (N_STATES*JOBS_PER_PE), sizeof(prob_t)*N_STATES*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_C[i], global_buf_C + i * (TRANS_SIZE*JOBS_PER_PE), sizeof(prob_t)*TRANS_SIZE*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_D[i], global_buf_D + i * (EMIT_SIZE*JOBS_PER_PE), sizeof(prob_t)*EMIT_SIZE*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
    #pragma HLS UNROLL
      copy_A(large_A[i], part_buf_A[i]);
    }
  }
  return;
}

void buffer_store(int flag, uint256_t* global_buf_A, state_t part_buf_A[UNROLL_FACTOR][N_OBS*JOBS_PER_PE]) {
#pragma HLS INLINE off
  uint256_t large_A[UNROLL_FACTOR][N_OBS*JOBS_PER_PE/32];
  #pragma HLS ARRAY_PARTITION variable=large_A dim=1 complete
  int i;
  if (flag) {
    for (i=0; i<UNROLL_FACTOR; i++) {
    #pragma HLS UNROLL
      copy_E(large_A[i], part_buf_A[i]);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(global_buf_A + i * (N_OBS*JOBS_PER_PE)/32, large_A[i], sizeof(state_t)*N_OBS*JOBS_PER_PE);
    }
  }
  return;
}

void buffer_compute(int flag,  tok_t local_obs[UNROLL_FACTOR][N_OBS*JOBS_PER_PE],
	                      prob_t local_init[UNROLL_FACTOR][N_STATES*JOBS_PER_PE],
		              prob_t local_transition[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE],      
                              prob_t local_emission[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE],
                             state_t local_path[UNROLL_FACTOR][N_OBS*JOBS_PER_PE]) {
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

void workload( uint256_t* obs, prob_t* init, prob_t* transition, prob_t* emission, uint256_t* path, int num_jobs ) {
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

    tok_t local_obs_x[UNROLL_FACTOR][N_OBS*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_obs_x complete dim=1
    tok_t local_obs_y[UNROLL_FACTOR][N_OBS*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_obs_y complete dim=1

    prob_t local_init_x[UNROLL_FACTOR][N_STATES*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_init_x complete dim=1
    prob_t local_init_y[UNROLL_FACTOR][N_STATES*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_init_y complete dim=1

    prob_t local_transition_x[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_transition_x complete dim=1
    prob_t local_transition_y[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_transition_y complete dim=1

    prob_t local_emission_x[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_emission_x complete dim=1
    prob_t local_emission_y[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_emission_y complete dim=1

    state_t local_path_x[UNROLL_FACTOR][N_OBS*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_path_x complete dim=1
    state_t local_path_y[UNROLL_FACTOR][N_OBS*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_path_y complete dim=1

    int num_batches = num_jobs / JOBS_PER_BATCH;

    int i;
    for (i=0; i<num_batches+2; i++) {
      int load_flag = i >= 0 && i < num_batches;
      int compute_flag = i >= 1 && i < num_batches+1;
      int store_flag = i >= 2 && i < num_batches+2;
      if (i % 2 == 0) {
        buffer_compute(compute_flag, local_obs_y, local_init_y, local_transition_y, local_emission_y, local_path_y);
        buffer_store(store_flag, path+(i-2)*JOBS_PER_BATCH*N_OBS/32, local_path_x);
        buffer_load(load_flag, obs+i*JOBS_PER_BATCH*N_OBS/32, local_obs_x, init+i*JOBS_PER_BATCH*N_STATES, local_init_x, transition+i*JOBS_PER_BATCH*TRANS_SIZE, local_transition_x, emission+i*JOBS_PER_BATCH*EMIT_SIZE, local_emission_x);
      } 
      else {
        buffer_compute(compute_flag, local_obs_x, local_init_x, local_transition_x, local_emission_x, local_path_x);
        buffer_store(store_flag, path+(i-2)*JOBS_PER_BATCH*N_OBS/32, local_path_y);
        buffer_load(load_flag, obs+i*JOBS_PER_BATCH*N_OBS/32, local_obs_y, init+i*JOBS_PER_BATCH*N_STATES, local_init_y, transition+i*JOBS_PER_BATCH*TRANS_SIZE, local_transition_y, emission+i*JOBS_PER_BATCH*EMIT_SIZE, local_emission_y);
      } 
    }
    return;
}

}
