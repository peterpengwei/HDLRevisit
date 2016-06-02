/*
*   Byte-oriented AES-256 implementation.
*   All lookup tables replaced with 'on the fly' calculations.
*/
#include "aes.h"
// #include <string.h>

#define F(x)   (((x)<<1) ^ ((((x)>>7) & 1) * 0x1b))
#define FD(x)  (((x) >> 1) ^ (((x) & 1) ? 0x8d : 0))

// #define BACK_TO_TABLES
// #ifdef BACK_TO_TABLES

// file scope constant array, to be set in a ROM
__constant unsigned char sbox[256] = {
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

// #else /* tableless subroutines */

// /* -------------------------------------------------------------------------- */
// unsigned char gf_alog(unsigned char x) // calculate anti-logarithm gen 3
// {
//     unsigned char atb = 1, z;

//     alog : while (x--) {z = atb; atb <<= 1; if (z & 0x80) atb^= 0x1b; atb ^= z;}

//     return atb;
// } /* gf_alog */

// /* -------------------------------------------------------------------------- */
// unsigned char gf_log(unsigned char x) // calculate logarithm gen 3
// {
//     unsigned char atb = 1, i = 0, z;

//     glog : do {
//         if (atb == x) break;
//         z = atb; atb <<= 1; if (z & 0x80) atb^= 0x1b; atb ^= z;
//     } while (++i > 0);

//     return i;
// } /* gf_log */


// /* -------------------------------------------------------------------------- */
// unsigned char gf_mulinv(unsigned char x) // calculate multiplicative inverse
// {
//     return (x) ? gf_alog(255 - gf_log(x)) : 0;
// } /* gf_mulinv */

// /* -------------------------------------------------------------------------- */
// unsigned char rj_sbox(unsigned char x)
// {
//     unsigned char y, sb;

//     sb = y = gf_mulinv(x);
//     y = (y<<1)|(y>>7); sb ^= y;  y = (y<<1)|(y>>7); sb ^= y;
//     y = (y<<1)|(y>>7); sb ^= y;  y = (y<<1)|(y>>7); sb ^= y;

//     return (sb ^ 0x63);
// } /* rj_sbox */
// #endif

/* -------------------------------------------------------------------------- */
// unsigned char rj_xtime(unsigned char x)
// {
//     return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
// } /* rj_xtime */

// /* -------------------------------------------------------------------------- */
// void aes_subBytes(unsigned char *buf)
// {
//     register unsigned char i = 16;

//     sub : while (i--) buf[i] = rj_sbox(buf[i]);
// } /* aes_subBytes */

// /* -------------------------------------------------------------------------- */
// void aes_addRoundKey(unsigned char *buf, unsigned char *key)
// {
//     register unsigned char i = 16;

//     addkey : while (i--) buf[i] ^= key[i];
// } /* aes_addRoundKey */

// /* -------------------------------------------------------------------------- */
// void aes_addRoundKey_cpy(unsigned char *buf, unsigned char *key, unsigned char *cpk)
// {
//     register unsigned char i = 16;

//     cpkey : while (i--)  buf[i] ^= (cpk[i] = key[i]), cpk[16+i] = key[16 + i];
// } /* aes_addRoundKey_cpy */


// /* -------------------------------------------------------------------------- */
// void aes_shiftRows(unsigned char *buf)
// {
//     register unsigned char i, j; /* to make it potentially parallelable :) */

//     i = buf[1]; buf[1] = buf[5]; buf[5] = buf[9]; buf[9] = buf[13]; buf[13] = i;
//     i = buf[10]; buf[10] = buf[2]; buf[2] = i;
//     j = buf[3]; buf[3] = buf[15]; buf[15] = buf[11]; buf[11] = buf[7]; buf[7] = j;
//     j = buf[14]; buf[14] = buf[6]; buf[6]  = j;

// } /* aes_shiftRows */

// /* -------------------------------------------------------------------------- */
// void aes_mixColumns(unsigned char *buf)
// {
//     register unsigned char i, a, b, c, d, e;

//     mix : for (i = 0; i < 16; i += 4)
//     {
//         a = buf[i]; b = buf[i + 1]; c = buf[i + 2]; d = buf[i + 3];
//         e = a ^ b ^ c ^ d;
//         buf[i] ^= e ^ rj_xtime(a^b);   buf[i+1] ^= e ^ rj_xtime(b^c);
//         buf[i+2] ^= e ^ rj_xtime(c^d); buf[i+3] ^= e ^ rj_xtime(d^a);
//     }
// } /* aes_mixColumns */

// /* -------------------------------------------------------------------------- */
// void aes_expandEncKey(unsigned char *k, unsigned char *rc)
// {
//     register unsigned char i;

//     k[0] ^= rj_sbox(k[29]) ^ (*rc);
//     k[1] ^= rj_sbox(k[30]);
//     k[2] ^= rj_sbox(k[31]);
//     k[3] ^= rj_sbox(k[28]);
//     *rc = F( *rc);

//     exp1 : for(i = 4; i < 16; i += 4)  k[i] ^= k[i-4],   k[i+1] ^= k[i-3],
//         k[i+2] ^= k[i-2], k[i+3] ^= k[i-1];
//     k[16] ^= rj_sbox(k[12]);
//     k[17] ^= rj_sbox(k[13]);
//     k[18] ^= rj_sbox(k[14]);
//     k[19] ^= rj_sbox(k[15]);

//     exp2 : for(i = 20; i < 32; i += 4) k[i] ^= k[i-4],   k[i+1] ^= k[i-3],
//         k[i+2] ^= k[i-2], k[i+3] ^= k[i-1];

// } /* aes_expandEncKey */

/* -------------------------------------------------------------------------- */
// void aes256_encrypt_ecb(aes256_context *ctx, unsigned char* k, unsigned char* buf)
//void aes256_encrypt_ecb(aes256_context *ctx, unsigned char k[32], unsigned char buf[16])
__kernel void
__attribute__((task))
workload( __global unsigned char * restrict k, 
          __global unsigned char * restrict buf)
{
    //INIT
    aes256_context ctx;
    unsigned char rcon = 1;
    unsigned char i, j;
    unsigned char p, q;

    ecb1 : for (i = 0; i < sizeof(ctx.key); i++) {
        ctx.enckey[i] = ctx.deckey[i] = k[i];
    }
    ecb2 : for (j = 8; --j;) {
        // aes_expandEncKey(ctx.deckey, &rcon);

        ctx.deckey[0] ^= rj_sbox(ctx.deckey[29]) ^ rcon;
        ctx.deckey[1] ^= rj_sbox(ctx.deckey[30]);
        ctx.deckey[2] ^= rj_sbox(ctx.deckey[31]);
        ctx.deckey[3] ^= rj_sbox(ctx.deckey[28]);
        rcon = F(rcon);

        exp11 : for (i = 4; i < 16; i += 4)  
            ctx.deckey[i] ^= ctx.deckey[i-4],   
            ctx.deckey[i+1] ^= ctx.deckey[i-3], 
            ctx.deckey[i+2] ^= ctx.deckey[i-2], 
            ctx.deckey[i+3] ^= ctx.deckey[i-1];
        
        ctx.deckey[16] ^= rj_sbox(ctx.deckey[12]);
        ctx.deckey[17] ^= rj_sbox(ctx.deckey[13]);
        ctx.deckey[18] ^= rj_sbox(ctx.deckey[14]);
        ctx.deckey[19] ^= rj_sbox(ctx.deckey[15]);

        exp12 : for (i = 20; i < 32; i += 4) 
            ctx.deckey[i] ^= ctx.deckey[i-4],   
            ctx.deckey[i+1] ^= ctx.deckey[i-3],
            ctx.deckey[i+2] ^= ctx.deckey[i-2], 
            ctx.deckey[i+3] ^= ctx.deckey[i-1];
    }

    //DEC
    // aes_addRoundKey_cpy(buf, ctx.enckey, ctx.key);
    i = 16;
    cpkey1 : while (i--) buf[i] ^= (ctx.key[i] = ctx.enckey[i]), ctx.key[16+i] = ctx.enckey[16 + i];

    ecb3 : for (j = 1, rcon = 1; j < 14; ++j) {
        // aes_subBytes(buf);
        i = 16;
        sub1 : while (i--) buf[i] = rj_sbox(buf[i]);

        // aes_shiftRows(buf);
        /* to make it potentially parallelable :) */
        p = buf[1]; buf[1] = buf[5]; buf[5] = buf[9]; buf[9] = buf[13]; buf[13] = p;
        p = buf[10]; buf[10] = buf[2]; buf[2] = p;
        q = buf[3]; buf[3] = buf[15]; buf[15] = buf[11]; buf[11] = buf[7]; buf[7] = q;
        q = buf[14]; buf[14] = buf[6]; buf[6]  = q;

        // aes_mixColumns(buf);
        unsigned char a, b, c, d, e;
        unsigned char ab, bc, cd, da;
        mix : for (i = 0; i < 16; i += 4) {
            a = buf[i]; b = buf[i + 1]; c = buf[i + 2]; d = buf[i + 3];
            e = a ^ b ^ c ^ d;
            ab = ((a^b) & 0x80) ? (((a^b) << 1) ^ 0x1b) : ((a^b) << 1);
            bc = ((b^c) & 0x80) ? (((b^c) << 1) ^ 0x1b) : ((b^c) << 1);
            cd = ((c^d) & 0x80) ? (((c^d) << 1) ^ 0x1b) : ((c^d) << 1);
            da = ((d^a) & 0x80) ? (((d^a) << 1) ^ 0x1b) : ((d^a) << 1);
            buf[i] ^= e ^ ab;   buf[i+1] ^= e ^ bc;
            buf[i+2] ^= e ^ cd; buf[i+3] ^= e ^ da;
        }

        if ( j & 1 ) {
            // aes_addRoundKey( buf, &ctx.key[16]);
            i = 16;
            addkey1 : while (i--) buf[i] ^= ctx.key[i+16];
        } else {
            // aes_expandEncKey(ctx.key, &rcon);
            ctx.key[0] ^= rj_sbox(ctx.key[29]) ^ rcon;
            ctx.key[1] ^= rj_sbox(ctx.key[30]);
            ctx.key[2] ^= rj_sbox(ctx.key[31]);
            ctx.key[3] ^= rj_sbox(ctx.key[28]);
            rcon = F(rcon);

            exp21 : for (i = 4; i < 16; i += 4)  
                ctx.key[i]   ^= ctx.key[i-4],   
                ctx.key[i+1] ^= ctx.key[i-3],
                ctx.key[i+2] ^= ctx.key[i-2], 
                ctx.key[i+3] ^= ctx.key[i-1];
            ctx.key[16] ^= rj_sbox(ctx.key[12]);
            ctx.key[17] ^= rj_sbox(ctx.key[13]);
            ctx.key[18] ^= rj_sbox(ctx.key[14]);
            ctx.key[19] ^= rj_sbox(ctx.key[15]);

            exp22 : for (i = 20; i < 32; i += 4) 
                ctx.key[i] ^= ctx.key[i-4],   ctx.key[i+1] ^= ctx.key[i-3],
                ctx.key[i+2] ^= ctx.key[i-2], ctx.key[i+3] ^= ctx.key[i-1];

            // aes_addRoundKey(buf, ctx.key);
            i = 16;
            addkey2 : while (i--) buf[i] ^= ctx.key[i];
        }
    }
    // aes_subBytes(buf);
    i = 16;
    sub2 : while (i--) buf[i] = rj_sbox(buf[i]);

    // aes_shiftRows(buf);
    /* to make it potentially parallelable :) */
    p = buf[1]; buf[1] = buf[5]; buf[5] = buf[9]; buf[9] = buf[13]; buf[13] = p;
    p = buf[10]; buf[10] = buf[2]; buf[2] = p;
    q = buf[3]; buf[3] = buf[15]; buf[15] = buf[11]; buf[11] = buf[7]; buf[7] = q;
    q = buf[14]; buf[14] = buf[6]; buf[6]  = q;

    // aes_expandEncKey(ctx.key, &rcon);
    ctx.key[0] ^= rj_sbox(ctx.key[29]) ^ rcon;
    ctx.key[1] ^= rj_sbox(ctx.key[30]);
    ctx.key[2] ^= rj_sbox(ctx.key[31]);
    ctx.key[3] ^= rj_sbox(ctx.key[28]);
    rcon = F(rcon);

    exp31 : for (i = 4; i < 16; i += 4)  
        ctx.key[i]   ^= ctx.key[i-4],   
        ctx.key[i+1] ^= ctx.key[i-3],
        ctx.key[i+2] ^= ctx.key[i-2], 
        ctx.key[i+3] ^= ctx.key[i-1];
    ctx.key[16] ^= rj_sbox(ctx.key[12]);
    ctx.key[17] ^= rj_sbox(ctx.key[13]);
    ctx.key[18] ^= rj_sbox(ctx.key[14]);
    ctx.key[19] ^= rj_sbox(ctx.key[15]);

    exp32 : for (i = 20; i < 32; i += 4) 
        ctx.key[i] ^= ctx.key[i-4],   ctx.key[i+1] ^= ctx.key[i-3],
        ctx.key[i+2] ^= ctx.key[i-2], ctx.key[i+3] ^= ctx.key[i-1];

    // aes_addRoundKey(buf, ctx.key);
    i = 16;
    addkey3 : while (i--) buf[i] ^= ctx.key[i];
} /* aes256_encrypt */
