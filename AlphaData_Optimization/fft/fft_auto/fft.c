#include "fft.h"
#include <assert.h>

#define UNROLL_FACTOR 4
#define IMG_PER_BATCH 64
#define IMG_PER_PE ((IMG_PER_BATCH+UNROLL_FACTOR-1)/UNROLL_FACTOR)

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log, rootindex;
    double temp;
    log = 0;

    for(span=FFT_SIZE>>1; span; span>>=1, log++){
	int counter = 0;
        for(odd=span; odd<FFT_SIZE; odd++){
#pragma HLS loop_tripcount min=64 max=64 avg=64
#pragma HLS pipeline
	    counter++;
            odd |= span;
            even = odd ^ span;

            temp = real[even] + real[odd];
            real[odd] = real[even] - real[odd];
            real[even] = temp;

            temp = img[even] + img[odd];
            img[odd] = img[even] - img[odd];
            img[even] = temp;

            rootindex = (even<<log) & (FFT_SIZE - 1);
            if(rootindex){
                temp = real_twid[rootindex] * real[odd] -
                    img_twid[rootindex]  * img[odd];
                img[odd] = real_twid[rootindex]*img[odd] +
                    img_twid[rootindex]*real[odd];
                real[odd] = temp;
            }
        }
	printf ("counter = %d\n", counter);
    }
}

void pe(double real[][FFT_SIZE], double img[][FFT_SIZE], double real_twid[][FFT_SIZE/2], double img_twid[][FFT_SIZE/2], int num_strides) {
#pragma HLS inline off
    for (int i=0; i<IMG_PER_PE; i++) {
        if (i < num_strides) fft(real[i], img[i], real_twid[i], img_twid[i]);
    }
}

void compute(int flag, double real[][IMG_PER_PE][FFT_SIZE], double img[][IMG_PER_PE][FFT_SIZE],
		       double real_twid[][IMG_PER_PE][FFT_SIZE/2], double img_twid[][IMG_PER_PE][FFT_SIZE/2], int num_strides) {
#pragma HLS inline off
    if (flag) {
	for (int i=0; i<UNROLL_FACTOR; i++) {
#pragma HLS unroll
            int pe_strides = num_strides-i*IMG_PER_PE;
	    if (pe_strides > IMG_PER_PE) pe_strides = IMG_PER_PE;
	    if (pe_strides > 0) pe(real[i], img[i], real_twid[i], img_twid[i], pe_strides);
	}
    }
}

// void load(int flag, double local_real[][IMG_PER_PE][FFT_SIZE], double* real,
// 		    double local_img[][IMG_PER_PE][FFT_SIZE], double* img,
// 		    double local_real_twid[][IMG_PER_PE][FFT_SIZE/2], double* real_twid,
// 		    double local_img_twid[][IMG_PER_PE][FFT_SIZE/2], double* img_twid, int num_strides) {
void load(int flag, double* local_real, double* real,
		    double* local_img, double* img,
		    double* local_real_twid, double* real_twid,
		    double* local_img_twid, double* img_twid, int num_strides) {
#pragma HLS inline off
    if (flag) {
        memcpy(local_real, real, num_strides*FFT_SIZE*sizeof(double));
        memcpy(local_img, img, num_strides*FFT_SIZE*sizeof(double));
        memcpy(local_real_twid, real_twid, num_strides*FFT_SIZE/2*sizeof(double));
        memcpy(local_img_twid, img_twid, num_strides*FFT_SIZE/2*sizeof(double));
    }
}

void store(int flag, double* real, double local_real[][IMG_PER_PE][FFT_SIZE],
		     double* img, double local_img[][IMG_PER_PE][FFT_SIZE], int num_strides) {
#pragma HLS inline off
    if (flag) {
        memcpy(real, local_real, num_strides*FFT_SIZE*sizeof(double));
	memcpy(img, local_img, num_strides*FFT_SIZE*sizeof(double));
    }
}

void workload(double* real, double* img, double* real_twid, double* img_twid, int num_strides){
#pragma HLS INTERFACE m_axi port=real      offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=img       offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=real_twid offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=img_twid  offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=real      bundle=control
#pragma HLS INTERFACE s_axilite port=img       bundle=control
#pragma HLS INTERFACE s_axilite port=real_twid bundle=control
#pragma HLS INTERFACE s_axilite port=img_twid  bundle=control
#pragma HLS INTERFACE s_axilite port=num_strides bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    double local_real_x[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE];
#pragma HLS array_partition variable=local_real_x dim=1 complete
    double local_img_x[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE];
#pragma HLS array_partition variable=local_img_x dim=1 complete
    double local_real_twid_x[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE/2];
#pragma HLS array_partition variable=local_real_twid_x dim=1 complete
    double local_img_twid_x[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE/2];
#pragma HLS array_partition variable=local_img_twid_x dim=1 complete

    double local_real_y[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE];
#pragma HLS array_partition variable=local_real_y dim=1 complete
    double local_img_y[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE];
#pragma HLS array_partition variable=local_img_y dim=1 complete
    double local_real_twid_y[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE/2];
#pragma HLS array_partition variable=local_real_twid_y dim=1 complete
    double local_img_twid_y[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE/2];
#pragma HLS array_partition variable=local_img_twid_y dim=1 complete

    double local_real_z[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE];
#pragma HLS array_partition variable=local_real_z dim=1 complete
    double local_img_z[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE];
#pragma HLS array_partition variable=local_img_z dim=1 complete
    double local_real_twid_z[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE/2];
#pragma HLS array_partition variable=local_real_twid_z dim=1 complete
    double local_img_twid_z[UNROLL_FACTOR][IMG_PER_PE][FFT_SIZE/2];
#pragma HLS array_partition variable=local_img_twid_z dim=1 complete

    assert (num_strides == (1 << 18));

    int num_batches = (num_strides+IMG_PER_BATCH-1)/IMG_PER_BATCH;
    int tail_strides = num_strides % IMG_PER_BATCH;
    if (tail_strides == 0) tail_strides = IMG_PER_BATCH;

    for (int i=0; i<num_batches+2; i++) {
        int load_flag = i < num_batches;
	int compute_flag = i > 0 && i < num_batches+1;
	int store_flag = i > 1;
	int load_strides = i == num_batches-1? tail_strides:IMG_PER_BATCH;
	int compute_strides = i == num_batches? tail_strides:IMG_PER_BATCH;
	int store_strides = i == num_batches+1? tail_strides:IMG_PER_BATCH;

	if (i % 3 == 0) {

	    load(load_flag, local_real_x, real+i*IMG_PER_BATCH*FFT_SIZE,
			    local_img_x, img+i*IMG_PER_BATCH*FFT_SIZE,
			    local_real_twid_x, real_twid+i*IMG_PER_BATCH*FFT_SIZE/2,
			    local_img_twid_x, img_twid+i*IMG_PER_BATCH*FFT_SIZE/2, load_strides);
	    compute(compute_flag, local_real_z, local_img_z, local_real_twid_z, local_img_twid_z, compute_strides);
	    store(store_flag, real+(i-2)*IMG_PER_BATCH*FFT_SIZE, local_real_y,
			      img+(i-2)*IMG_PER_BATCH*FFT_SIZE, local_img_y, store_strides);
	}
	else if (i % 3 == 1) {
	    load(load_flag, local_real_y, real+i*IMG_PER_BATCH*FFT_SIZE,
			    local_img_y, img+i*IMG_PER_BATCH*FFT_SIZE,
			    local_real_twid_y, real_twid+i*IMG_PER_BATCH*FFT_SIZE/2,
			    local_img_twid_y, img_twid+i*IMG_PER_BATCH*FFT_SIZE/2, load_strides);
	    compute(compute_flag, local_real_x, local_img_x, local_real_twid_x, local_img_twid_x, compute_strides);
	    store(store_flag, real+(i-2)*IMG_PER_BATCH*FFT_SIZE, local_real_z,
			      img+(i-2)*IMG_PER_BATCH*FFT_SIZE, local_img_z, store_strides);
	}
	else {
	    load(load_flag, local_real_z, real+i*IMG_PER_BATCH*FFT_SIZE,
			    local_img_z, img+i*IMG_PER_BATCH*FFT_SIZE,
			    local_real_twid_z, real_twid+i*IMG_PER_BATCH*FFT_SIZE/2,
			    local_img_twid_z, img_twid+i*IMG_PER_BATCH*FFT_SIZE/2, load_strides);
	    compute(compute_flag, local_real_y, local_img_y, local_real_twid_y, local_img_twid_y, compute_strides);
	    store(store_flag, real+(i-2)*IMG_PER_BATCH*FFT_SIZE, local_real_x,
			      img+(i-2)*IMG_PER_BATCH*FFT_SIZE, local_img_x, store_strides);
	}

    }

    return;
}
