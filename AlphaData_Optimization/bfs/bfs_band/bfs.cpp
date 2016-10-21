#include "ap_int.h"
#define RANGE(var, h, l) (var).range(h, l)
/*
Implementation based on:
Hong, Oguntebi, Olukotun. "Efficient Parallel Graph Exploration on Multi-Core CPU and GPU." PACT, 2011.
*/
#include "bfs.h"
#define Q_PUSH(node) { queue[q_in==0?N_NODES-1:q_in-1]=node; q_in=(q_in+1)%N_NODES; }
#define Q_PEEK() (queue[q_out])
#define Q_POP() { q_out = (q_out+1)%N_NODES; }
#define Q_EMPTY() (q_in>q_out ? q_in==q_out+1 : (q_in==0)&&(q_out==N_NODES-1))
extern "C" {

void bfs(ap_uint<512> *edge_begin,ap_uint<512> *edge_end,ap_uint<512> *dst,uint32_t starting_node,ap_uint<512> *level,ap_uint<512> *level_counts)
{
  ap_uint<512> queue[256UL];
  uint32_t q_in;
  uint32_t q_out;
  uint32_t dummy;
  uint32_t n;
  uint32_t e;
  int i;
  const ap_int<512> init_level("0x7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f", 16);
  init_levels:
  for (i = 0; i < 64; i++) {
  #pragma HLS PIPELINE
    level[i] = init_level; 
  }
  init_horizons:
  for (i = 0; i < 2; i++) {
  #pragma HLS PIPELINE
    level_counts[i] = 0;
  }
  q_in = 1;
  q_out = 0;
  RANGE(level[starting_node / 64],starting_node % 64 * 8 + 7,starting_node % 64 * 8) = 0;
  RANGE(level_counts[0 / 16],0 % 16 * 32 + 31,0 % 16 * 32) = 1;
  {
    RANGE(queue[((q_in == ((unsigned int )0)?((unsigned int )(4096 - 1)) : q_in - ((unsigned int )1))) / 16],((q_in == ((unsigned int )0)?((unsigned int )(4096 - 1)) : q_in - ((unsigned int )1))) % 16 * 32 + 31,((q_in == ((unsigned int )0)?((unsigned int )(4096 - 1)) : q_in - ((unsigned int )1))) % 16 * 32) = starting_node;
    q_in = (q_in + 1) % 4096;
  }
  ;
  // Typically while(not_empty(queue)){
  loop_queue:
  for (dummy = 0; dummy < 4096; dummy++) {
    if (q_in > q_out?q_in == q_out + 1 : q_in == 0 && q_out == (4096 - 1)) 
      break; 
    n = RANGE(queue[q_out / 16],q_out % 16 * 32 + 31,q_out % 16 * 32);
    {
      q_out = (q_out + 1) % 4096;
    }
    ;
    uint32_t tmp_begin = RANGE(edge_begin[n / 16],n % 16 * 32 + 31,n % 16 * 32);
    uint32_t tmp_end = RANGE(edge_end[n / 16],n % 16 * 32 + 31,n % 16 * 32);
    loop_neighbors:
    for (e = tmp_begin; e < tmp_end; e++) {
      
      uint32_t tmp_dst = RANGE(dst[e / 16],e % 16 * 32 + 31,e % 16 * 32);
      uint8_t tmp_level = RANGE(level[tmp_dst / 64],tmp_dst % 64 * 8 + 7,tmp_dst % 64 * 8);
      // Unmarked
      if (tmp_level == 0x7f) {
        uint8_t tmp_level = ((RANGE(level[n / 64],n % 64 * 8 + 7,n % 64 * 8)) + 1);
        RANGE(level[tmp_dst / 64],tmp_dst % 64 * 8 + 7,tmp_dst % 64 * 8) = tmp_level;
        RANGE(level_counts[tmp_level / 16],tmp_level % 16 * 32 + 31,tmp_level % 16 * 32) = RANGE(level_counts[tmp_level / 16],tmp_level % 16 * 32 + 31,tmp_level % 16 * 32) + 1;
        {
          RANGE(queue[((q_in == ((unsigned int )0)?((unsigned int )(4096 - 1)) : q_in - ((unsigned int )1))) / 16],((q_in == ((unsigned int )0)?((unsigned int )(4096 - 1)) : q_in - ((unsigned int )1))) % 16 * 32 + 31,((q_in == ((unsigned int )0)?((unsigned int )(4096 - 1)) : q_in - ((unsigned int )1))) % 16 * 32) = tmp_dst;
          q_in = (q_in + 1) % 4096;
        }
        ;
      }
    }
  }
}

void workload(ap_uint<512> *edge_begin,ap_uint<512> *edge_end,ap_uint<512> *dst,ap_uint<512> *p_starting_node,ap_uint<512> *level,ap_uint<512> *level_counts)
{
  
#pragma HLS INTERFACE m_axi port=edge_begin offset=slave bundle=gmem
  
#pragma HLS INTERFACE m_axi port=edge_end offset=slave bundle=gmem
  
#pragma HLS INTERFACE m_axi port=dst offset=slave bundle=gmem
  
#pragma HLS INTERFACE m_axi port=p_starting_node offset=slave bundle=gmem
  
#pragma HLS INTERFACE m_axi port=level offset=slave bundle=gmem
  
#pragma HLS INTERFACE m_axi port=level_counts offset=slave bundle=gmem
  
#pragma HLS INTERFACE s_axilite port=edge_begin bundle=control
  
#pragma HLS INTERFACE s_axilite port=edge_end bundle=control
  
#pragma HLS INTERFACE s_axilite port=dst bundle=control
  
#pragma HLS INTERFACE s_axilite port=p_starting_node bundle=control
  
#pragma HLS INTERFACE s_axilite port=level bundle=control
  
#pragma HLS INTERFACE s_axilite port=level_counts bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control

  ap_uint<512> local_edge_begin[256UL];
  ap_uint<512> local_edge_end[256UL];
  ap_uint<512> local_dst[4096UL];
  ap_uint<512> local_level[64UL];
  ap_uint<512> local_level_counts[2UL];
  memcpy(local_edge_begin,edge_begin,sizeof(local_edge_begin));
  memcpy(local_edge_end,edge_end,sizeof(local_edge_end));
  memcpy(local_dst,dst,sizeof(local_dst));
  bfs(local_edge_begin,local_edge_end,local_dst, *p_starting_node,local_level,local_level_counts);
  memcpy(level,local_level,sizeof(local_level));
  memcpy(level_counts,local_level_counts,sizeof(local_level_counts));
  return ;
}

}
