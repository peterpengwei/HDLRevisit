#include "fft.h"

void fft(double real[SIZE], double img[SIZE], double real_twid[SIZE/2], double img_twid[SIZE/2]){
    int even, odd, span, log, rootindex;
    double temp;
    log = 0;

    for(span=SIZE>>1; span; span>>=1, log++){
        for(odd=span; odd<SIZE; odd++){
            odd |= span;
            even = odd ^ span;

            temp = real[even] + real[odd];
            real[odd] = real[even] - real[odd];
            real[even] = temp;

            temp = img[even] + img[odd];
            img[odd] = img[even] - img[odd];
            img[even] = temp;

            rootindex = (even<<log) & (SIZE - 1);
            if(rootindex){
                temp = real_twid[rootindex] * real[odd] -
                    img_twid[rootindex]  * img[odd];
                img[odd] = real_twid[rootindex]*img[odd] +
                    img_twid[rootindex]*real[odd];
                real[odd] = temp;
            }
        }
    }
}
void workload(double* real, double* img, double* real_twid, double* img_twid, int num_imgs){
#pragma HLS INTERFACE m_axi port=real      offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=img       offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=real_twid offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=img_twid  offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=real      bundle=control
#pragma HLS INTERFACE s_axilite port=img       bundle=control
#pragma HLS INTERFACE s_axilite port=real_twid bundle=control
#pragma HLS INTERFACE s_axilite port=img_twid  bundle=control
#pragma HLS INTERFACE s_axilite port=num_imgs  bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    for (int i=0; i<num_imgs; i++) {
	fft(real+i*SIZE, img+i*SIZE, real_twid+i*SIZE/2, img_twid+i*SIZE/2);
    }
    return;
}
