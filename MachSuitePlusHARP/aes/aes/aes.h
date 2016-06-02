/*
*   Byte-oriented AES-256 implementation.
*   All lookup tables replaced with 'on the fly' calculations.
*/

// #include <inttypes.h>

typedef struct {
  unsigned char key[32];
  unsigned char enckey[32];
  unsigned char deckey[32];
} aes256_context;

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  aes256_context ctx;
  unsigned char k[32];
  unsigned char buf[16];
};
