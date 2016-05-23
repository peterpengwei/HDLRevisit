#define twoPI 6.28318530717959

#define SIZE 1024

void fft(double real[SIZE], double img[SIZE], 
	double real_twid[SIZE/2], double img_twid[SIZE/2]);

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
        double real[SIZE];
        double img[SIZE];
        double real_twid[SIZE/2];
        double img_twid[SIZE/2];
};
