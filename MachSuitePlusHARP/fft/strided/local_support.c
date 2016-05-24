#include "fft.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON ((double)1.0e-6)

void run_benchmark( void *vargs, cl_context& context, 
  cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  static unsigned * real_buffer = 
    (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->real), 1024);
  static unsigned * img_buffer = 
    (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->img), 1024);
  static unsigned * real_twid_buffer = 
    (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->real_twid), 1024);
  static unsigned * img_twid_buffer = 
    (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->img_twid), 1024);

  // Write our data set into device buffers  
  //
  memcpy(real_buffer, args->real, sizeof(args->real));
  memcpy(img_buffer, args->img, sizeof(args->img));
  memcpy(real_twid_buffer, args->real_twid, sizeof(args->real_twid));
  memcpy(img_twid_buffer, args->img_twid, sizeof(args->img_twid));

  // Set the arguments to our compute kernel
  //
  int status;
  status  = clSetKernelArgSVMPointerAltera(kernel, 0, (void*)real_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 1, (void*)img_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 2, (void*)real_twid_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 3, (void*)img_twid_buffer);
  if (status != CL_SUCCESS) {
    dump_error("Failed set args.", status);
    exit(1);
  }

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //

#ifdef OPENCL_KERNEL
  status = clEnqueueTask(commands, kernel, 0, NULL, NULL);
#else
  printf("Error: C kernel is not currently supported!\n");
  exit(1);
#endif
  if (status)
  {
    printf("Error: Failed to execute kernel! %d\n", status);
    printf("Test failed\n");
    exit(1);
  }
  clFinish(commands);

  // Read back the results from the device to verify the output
  //
  memcpy(args->real, real_buffer, sizeof(args->real));
  memcpy(args->img, img_buffer, sizeof(args->img));
}

/* Input format:
%% Section 1
double: signal (real part)
%% Section 2
double: signal (complex part)
%% Section 3
double: twiddle factor (real part)
%% Section 4
double: twiddle factor (complex part)
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_double_array(s, data->real, SIZE);

  s = find_section_start(p,2);
  parse_double_array(s, data->img, SIZE);

  s = find_section_start(p,3);
  parse_double_array(s, data->real_twid, SIZE/2);

  s = find_section_start(p,4);
  parse_double_array(s, data->img_twid, SIZE/2);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_double_array(fd, data->real, SIZE);

  write_section_header(fd);
  write_double_array(fd, data->img, SIZE);

  write_section_header(fd);
  write_double_array(fd, data->real_twid, SIZE/2);

  write_section_header(fd);
  write_double_array(fd, data->img_twid, SIZE/2);
}

/* Output format:
%% Section 1
double: freq (real part)
%% Section 2
double: freq (complex part)
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_double_array(s, data->real, SIZE);

  s = find_section_start(p,2);
  parse_double_array(s, data->img, SIZE);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_double_array(fd, data->real, SIZE);

  write_section_header(fd);
  write_double_array(fd, data->img, SIZE);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i;
  double real_diff, img_diff;

  for(i=0; i<SIZE; i++) {
    real_diff = data->real[i] - ref->real[i];
    img_diff = data->img[i] - ref->img[i];
    has_errors |= (real_diff<-EPSILON) || (EPSILON<real_diff);
    //if( has_errors )
      //printf("%d (real): %f (%f)\n", i, real_diff, EPSILON);
    has_errors |= (img_diff<-EPSILON) || (EPSILON<img_diff);
    //if( has_errors )
      //printf("%d (img): %f (%f)\n", i, img_diff, EPSILON);
  }

  // Return true if it's correct.
  return !has_errors;
}
