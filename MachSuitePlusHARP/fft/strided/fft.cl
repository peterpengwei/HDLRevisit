#include "fft.h"

__kernel void
__attribute__((task))
workload( __global double * restrict real, 
          __global double * restrict img,
          __global double * restrict real_twid,
          __global double * restrict img_twid) {
    int even, odd, span, log, rootindex;
    double temp;
    log = 0;

    for (span = SIZE>>1; span; span>>=1, log++) {
        for (odd = span; odd < SIZE; odd++) {
            odd |= span;
            even = odd ^ span;

            temp = real[even] + real[odd];
            real[odd] = real[even] - real[odd];
            real[even] = temp;

            temp = img[even] + img[odd];
            img[odd] = img[even] - img[odd];
            img[even] = temp;

            rootindex = (even<<log) & (SIZE - 1);
            if (rootindex) {
                temp = real_twid[rootindex] * real[odd] -
                    img_twid[rootindex]  * img[odd];
                img[odd] = real_twid[rootindex]*img[odd] +
                    img_twid[rootindex]*real[odd];
                real[odd] = temp;
            }
        }
    }
}
