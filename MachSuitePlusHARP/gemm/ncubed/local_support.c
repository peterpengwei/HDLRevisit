#include "gemm.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON ((TYPE)1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  static unsigned *m1_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->m1), 1024); 
  static unsigned *m2_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->m2), 1024); 
  static unsigned *prod_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->prod), 1024); 

  // Write our data set into device buffers  
  //
  memcpy(m1_buffer, args->m1, sizeof(args->m1));
  memcpy(m2_buffer, args->m2, sizeof(args->m2));
    
  // Set the arguments to our compute kernel
  //
  int status;
  status = clSetKernelArgSVMPointerAltera(kernel, 0, (void*)m1_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 1, (void*)m2_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 2, (void*)prod_buffer);
  if(status != CL_SUCCESS) {
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
  memcpy(args->prod, prod_buffer, sizeof(args->prod));
}

/* Input format:
%% Section 1
TYPE[N]: matrix 1
%% Section 2
TYPE[N]: matrix 2
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->m1, N);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->m2, N);

}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->m1, N);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->m2, N);
}

/* Output format:
%% Section 1
TYPE[N]: output matrix
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->prod, N);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->prod, N);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int r,c;
  TYPE diff;

  for( r=0; r<row_size; r++ ) {
    for( c=0; c<col_size; c++ ) {
      diff = data->prod[r*row_size + c] - ref->prod[r*row_size+c];
      has_errors |= (diff<-EPSILON) || (EPSILON<diff);
    }
  }

  // Return true if it's correct.
  return !has_errors;
}
