/*
*   Byte-oriented AES-256 implementation.
*   All lookup tables replaced with 'on the fly' calculations.
*/
#include "aes.h"
#include <string.h>
#include "ap_int.h"
#include <iostream>

typedef ap_uint<128> uint128_t;
typedef ap_uint<256> uint256_t;
typedef ap_uint<512> uint512_t;

typedef struct {
    uint256_t key;
    uint256_t enckey;
    uint256_t deckey;
} aes_ctx;

extern "C" {

#define F(x)   (((x)<<1) ^ ((((x)>>7) & 1) * 0x1b))
#define FD(x)  (((x) >> 1) ^ (((x) & 1) ? 0x8d : 0))

#define BACK_TO_TABLES
#ifdef BACK_TO_TABLES

#define BUF_SIZE_OFFSET 16
#define BUF_SIZE ((1) << (BUF_SIZE_OFFSET))

#define UNROLL_FACTOR 16

const uint8_t sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

#define rj_sbox(x)     sbox[(x)]

#else /* tableless subroutines */

/* -------------------------------------------------------------------------- */
uint8_t gf_alog(uint8_t x) // calculate anti-logarithm gen 3
{
    uint8_t atb = 1, z;

    alog : while (x--) {z = atb; atb <<= 1; if (z & 0x80) atb^= 0x1b; atb ^= z;}

    return atb;
} /* gf_alog */

/* -------------------------------------------------------------------------- */
uint8_t gf_log(uint8_t x) // calculate logarithm gen 3
{
    uint8_t atb = 1, i = 0, z;

    glog : do {
        if (atb == x) break;
        z = atb; atb <<= 1; if (z & 0x80) atb^= 0x1b; atb ^= z;
    } while (++i > 0);

    return i;
} /* gf_log */


/* -------------------------------------------------------------------------- */
uint8_t gf_mulinv(uint8_t x) // calculate multiplicative inverse
{
    return (x) ? gf_alog(255 - gf_log(x)) : 0;
} /* gf_mulinv */

/* -------------------------------------------------------------------------- */
uint8_t rj_sbox(uint8_t x)
{
    uint8_t y, sb;

    sb = y = gf_mulinv(x);
    y = (y<<1)|(y>>7); sb ^= y;  y = (y<<1)|(y>>7); sb ^= y;
    y = (y<<1)|(y>>7); sb ^= y;  y = (y<<1)|(y>>7); sb ^= y;

    return (sb ^ 0x63);
} /* rj_sbox */
#endif

/* -------------------------------------------------------------------------- */
uint8_t rj_xtime(uint8_t x)
{
    return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
} /* rj_xtime */

/* -------------------------------------------------------------------------- */
void aes_subBytes(uint128_t *buf)
{
    register uint8_t i = 16;

    sub : while (i--) {
    #pragma HLS UNROLL
      (*buf)(i*8+7, i*8) = rj_sbox((*buf)(i*8+7, i*8));
    }
} /* aes_subBytes */

/* -------------------------------------------------------------------------- */
void aes_addRoundKey(uint128_t *buf, uint128_t key)
{
    register uint8_t i = 16;

    addkey : while (i--) {
    #pragma HLS UNROLL
	(*buf)(i*8+7, i*8) = (*buf)(i*8+7, i*8) ^ (key)(i*8+7, i*8);
    }
} /* aes_addRoundKey */

/* -------------------------------------------------------------------------- */
void aes_addRoundKey_cpy(uint128_t *buf, uint256_t *key, uint256_t *cpk)
{
    register uint8_t i = 16;

    cpkey : while (i--)  {
    #pragma HLS UNROLL
        (*cpk)(i*8+7, i*8) = (*key)(i*8+7, i*8);
        (*buf)(i*8+7, i*8) = (*buf)(i*8+7, i*8) ^ ((*cpk)(i*8+7, i*8));
	(*cpk)((16+i)*8+7, (16+i)*8) = (*key)((16+i)*8+7, (16+i)*8);
    }
} /* aes_addRoundKey_cpy */


/* -------------------------------------------------------------------------- */
void aes_shiftRows(uint128_t *buf)
{
    register uint8_t i, j; /* to make it potentially parallelable :) */

    i = (*buf)(1*8+7, 1*8); (*buf)(1*8+7, 1*8) = (*buf)(5*8+7, 5*8); (*buf)(5*8+7, 5*8) = (*buf)(9*8+7, 9*8); (*buf)(9*8+7, 9*8) = (*buf)(13*8+7, 13*8); (*buf)(13*8+7, 13*8) = i;
    i = (*buf)(10*8+7, 10*8); (*buf)(10*8+7, 10*8) = (*buf)(2*8+7, 2*8); (*buf)(2*8+7, 2*8) = i;
    j = (*buf)(3*8+7, 3*8); (*buf)(3*8+7, 3*8) = (*buf)(15*8+7, 15*8); (*buf)(15*8+7, 15*8) = (*buf)(11*8+7, 11*8); (*buf)(11*8+7, 11*8) = (*buf)(7*8+7, 7*8); (*buf)(7*8+7, 7*8) = j;
    j = (*buf)(14*8+7, 14*8); (*buf)(14*8+7, 14*8) = (*buf)(6*8+7, 6*8); (*buf)(6*8+7, 6*8)  = j;

} /* aes_shiftRows */

/* -------------------------------------------------------------------------- */
void aes_mixColumns(uint128_t *buf)
{
    register uint8_t i, a, b, c, d, e;

    mix : for (i = 0; i < 16; i += 4)
    {
    #pragma HLS UNROLL
        a = (*buf)(i*8+7, i*8); 
	b = (*buf)((i + 1)*8+7, (i + 1)*8); 
	c = (*buf)((i + 2)*8+7, (i + 2)*8); 
	d = (*buf)((i + 3)*8+7, (i + 3)*8);
        e = a ^ b ^ c ^ d;
        (*buf)(i*8+7, i*8)             = (*buf)(i*8+7, i*8)             ^ (e ^ rj_xtime(a^b));   
	(*buf)((i + 1)*8+7, (i + 1)*8) = (*buf)((i + 1)*8+7, (i + 1)*8) ^ (e ^ rj_xtime(b^c));
        (*buf)((i + 2)*8+7, (i + 2)*8) = (*buf)((i + 2)*8+7, (i + 2)*8) ^ (e ^ rj_xtime(c^d)); 
	(*buf)((i + 3)*8+7, (i + 3)*8) = (*buf)((i + 3)*8+7, (i + 3)*8) ^ (e ^ rj_xtime(d^a));
    }
} /* aes_mixColumns */

/* -------------------------------------------------------------------------- */
void aes_expandEncKey(uint256_t *k, uint8_t *rc)
{
    register uint8_t i;

    (*k)(0*8+7, 0*8) = (*k)(0*8+7, 0*8) ^ (rj_sbox((*k)(29*8+7, 29*8)) ^ (*rc));
    (*k)(1*8+7, 1*8) = (*k)(1*8+7, 1*8) ^ (rj_sbox((*k)(30*8+7, 30*8))        );
    (*k)(2*8+7, 2*8) = (*k)(2*8+7, 2*8) ^ (rj_sbox((*k)(31*8+7, 31*8))        );
    (*k)(3*8+7, 3*8) = (*k)(3*8+7, 3*8) ^ (rj_sbox((*k)(28*8+7, 28*8))        );
    *rc = F( *rc);

    exp1 : for(i = 4; i < 16; i += 4) {
    #pragma HLS UNROLL
	(*k)(i*8+7, i*8)         = (*k)(i*8+7, i*8)          ^ (*k)((i-4)*8+7, (i-4)*8);
     	(*k)((i+1)*8+7, (i+1)*8) = (*k)((i+1)*8+7, (i+1)*8)  ^ (*k)((i-3)*8+7, (i-3)*8);
        (*k)((i+2)*8+7, (i+2)*8) = (*k)((i+2)*8+7, (i+2)*8)  ^ (*k)((i-2)*8+7, (i-2)*8); 
	(*k)((i+3)*8+7, (i+3)*8) = (*k)((i+3)*8+7, (i+3)*8)  ^ (*k)((i-1)*8+7, (i-1)*8);
    }
    (*k)(16*8+7, 16*8) = (*k)(16*8+7, 16*8) ^ rj_sbox((*k)(12*8+7, 12*8));
    (*k)(17*8+7, 17*8) = (*k)(17*8+7, 17*8) ^ rj_sbox((*k)(13*8+7, 13*8));
    (*k)(18*8+7, 18*8) = (*k)(18*8+7, 18*8) ^ rj_sbox((*k)(14*8+7, 14*8));
    (*k)(19*8+7, 19*8) = (*k)(19*8+7, 19*8) ^ rj_sbox((*k)(15*8+7, 15*8));

    exp2 : for(i = 20; i < 32; i += 4) {
    #pragma HLS UNROLL
	(*k)(i*8+7, i*8)         = (*k)(i*8+7, i*8)         ^ (*k)((i-4)*8+7, (i-4)*8);
     	(*k)((i+1)*8+7, (i+1)*8) = (*k)((i+1)*8+7, (i+1)*8) ^ (*k)((i-3)*8+7, (i-3)*8);
        (*k)((i+2)*8+7, (i+2)*8) = (*k)((i+2)*8+7, (i+2)*8) ^ (*k)((i-2)*8+7, (i-2)*8); 
	(*k)((i+3)*8+7, (i+3)*8) = (*k)((i+3)*8+7, (i+3)*8) ^ (*k)((i-1)*8+7, (i-1)*8);
    }

} /* aes_expandEncKey */

/* -------------------------------------------------------------------------- */
void aes256_encrypt_ecb(uint256_t* k, uint128_t* buf)
{
    aes_ctx ctx_body;
    aes_ctx* ctx = &ctx_body;
    //INIT
    uint8_t rcon = 1;
    uint8_t i;

    ctx->enckey = ctx->deckey = *k;

    ecb2 : for (i = 8;--i;){
        aes_expandEncKey(&(ctx->deckey), &rcon);
    }

    //DEC
    aes_addRoundKey_cpy(buf, &(ctx->enckey), &(ctx->key));
    ecb3 : for(i = 1, rcon = 1; i < 14; ++i)
    {
        aes_subBytes(buf);
        aes_shiftRows(buf);
        aes_mixColumns(buf);
        if( i & 1 ) aes_addRoundKey( buf, (ctx->key)(255, 128));
        else aes_expandEncKey(&ctx->key, &rcon), aes_addRoundKey(buf, (ctx->key)(127, 0));
    }
    aes_subBytes(buf);
    aes_shiftRows(buf);
    aes_expandEncKey(&ctx->key, &rcon);
    aes_addRoundKey(buf, (ctx->key)(127, 0));
} /* aes256_encrypt */

void aes_cacheline(uint256_t* local_key, uint512_t* buf) {
    int i;
    uint128_t value[4];
    #pragma HLS ARRAY_PARTITION variable=value complete dim=1
    value[0] = (*buf)(128*0+127, 128*0);
    value[1] = (*buf)(128*1+127, 128*1);
    value[2] = (*buf)(128*2+127, 128*2);
    value[3] = (*buf)(128*3+127, 128*3);
    aes256_encrypt_ecb(local_key, &value[0]);
    aes256_encrypt_ecb(local_key, &value[1]);
    aes256_encrypt_ecb(local_key, &value[2]);
    aes256_encrypt_ecb(local_key, &value[3]);
    (*buf)(128*0+127, 128*0) = value[0];
    (*buf)(128*1+127, 128*1) = value[1];
    (*buf)(128*2+127, 128*2) = value[2];
    (*buf)(128*3+127, 128*3) = value[3];
}

void aes_tiling(uint256_t* local_key, uint512_t* buf) {
    for (int k=0; k<BUF_SIZE/UNROLL_FACTOR/64; k++) {
        aes_cacheline(local_key, &buf[k]);
    }
}

void buffer_load(int flag, int size, uint512_t* global_buf, uint512_t part_buf[UNROLL_FACTOR][BUF_SIZE/UNROLL_FACTOR/64]) {
#pragma HLS INLINE off
  if (flag) {
    for (int i=0; i<UNROLL_FACTOR; i++) {
    #pragma HLS UNROLL
      memcpy(part_buf[i], global_buf + i * (BUF_SIZE/UNROLL_FACTOR/64), BUF_SIZE/UNROLL_FACTOR);
    }
  }
  return;
}

void buffer_store(int flag, int size, uint512_t* global_buf, uint512_t part_buf[UNROLL_FACTOR][BUF_SIZE/UNROLL_FACTOR/64]) {
#pragma HLS INLINE off
  if (flag) {
    for (int i=0; i<UNROLL_FACTOR; i++) {
    #pragma HLS UNROLL
      memcpy(global_buf + i * (BUF_SIZE/UNROLL_FACTOR/64), part_buf[i], BUF_SIZE/UNROLL_FACTOR);
    }
  }
  return;
}

void buffer_compute(int flag, uint512_t buf[UNROLL_FACTOR][BUF_SIZE/UNROLL_FACTOR/64], int size, uint256_t* key) {
#pragma HLS INLINE off
  if (flag) {
    uint256_t local_key[UNROLL_FACTOR];
    #pragma HLS ARRAY_PARTITION variable=local_key cyclic factor=16 dim=1

    int i,j,k;
    for (i=0; i<UNROLL_FACTOR; i++) {
    #pragma HLS UNROLL
        local_key[i] = *key;
    }

    unroll_loop: for (j=0; j<UNROLL_FACTOR; j++) {
    #pragma HLS UNROLL
        aes_tiling(&local_key[j], buf[j]);
    }
  }
  return;
}

void workload(uint256_t* key, uint512_t* a, int data_size) {
#pragma HLS INTERFACE m_axi port=key offset=slave bundle=gmem1 
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem2 
#pragma HLS INTERFACE s_axilite port=key bundle=control
#pragma HLS INTERFACE s_axilite port=a bundle=control 
#pragma HLS INTERFACE s_axilite port=data_size bundle=control 
#pragma HLS INTERFACE s_axilite port=return bundle=control 
  int num_batches = data_size / BUF_SIZE;

  uint512_t buf_partition_x[UNROLL_FACTOR][BUF_SIZE/UNROLL_FACTOR/64];
  #pragma HLS ARRAY_PARTITION variable=buf_partition_x cyclic factor=16 dim=1
  uint512_t buf_partition_y[UNROLL_FACTOR][BUF_SIZE/UNROLL_FACTOR/64];
  #pragma HLS ARRAY_PARTITION variable=buf_partition_y cyclic factor=16 dim=1
  uint512_t buf_partition_z[UNROLL_FACTOR][BUF_SIZE/UNROLL_FACTOR/64];
  #pragma HLS ARRAY_PARTITION variable=buf_partition_z cyclic factor=16 dim=1
  
  uint256_t local_key = *key;

  int i;
  for (i=0; i<num_batches+2; i++) {
    int load_flag = i >= 0 && i < num_batches;
    int compute_flag = i >= 1 && i < num_batches+1;
    int store_flag = i >= 2 && i < num_batches+2;
    int load_size = BUF_SIZE;
    int compute_size = BUF_SIZE;
    int store_size = BUF_SIZE;
    if (i % 3 == 0) {
      buffer_load(load_flag, load_size, a+i*BUF_SIZE/64, buf_partition_x);
      buffer_compute(compute_flag, buf_partition_z, compute_size, &local_key);
      buffer_store(store_flag, store_size, a+(i-2)*BUF_SIZE/64, buf_partition_y);
    } 
    else if (i % 3 == 1) {
      buffer_load(load_flag, load_size, a+i*BUF_SIZE/64, buf_partition_y);
      buffer_compute(compute_flag, buf_partition_x, compute_size, &local_key);
      buffer_store(store_flag, store_size, a+(i-2)*BUF_SIZE/64, buf_partition_z);
    } 
    else {
      buffer_load(load_flag, load_size, a+i*BUF_SIZE/64, buf_partition_z);
      buffer_compute(compute_flag, buf_partition_y, compute_size, &local_key);
      buffer_store(store_flag, store_size, a+(i-2)*BUF_SIZE/64, buf_partition_x);
    } 
  }
  return;
}

}
