#include "stencil.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON (1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  static unsigned *orig_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->orig), 1024); 
  static unsigned *sol_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->sol), 1024); 
  static unsigned *C_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->C), 1024); 

  // Write our data set into device buffers  
  //
  memcpy(orig_buffer, args->orig, sizeof(args->orig));
  memcpy(sol_buffer, args->sol, sizeof(args->sol));
  memcpy(C_buffer, args->C, sizeof(args->C));
    
  // Set the arguments to our compute kernel
  //
  int status;
  status = clSetKernelArgSVMPointerAltera(kernel, 0, (void*)C_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 1, (void*)orig_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 2, (void*)sol_buffer);
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
  memcpy(args->sol, sol_buffer, sizeof(args->sol));
}

/* Input format:
%% Section 1
TYPE[2]: stencil coefficients (inner/outer)
%% Section 2
TYPE[SIZE]: input matrix
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->C, 2);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->orig, SIZE);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->C, 2);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->orig, SIZE);
}

/* Output format:
%% Section 1
TYPE[SIZE]: solution matrix
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->sol, SIZE);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->sol, SIZE);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i;
  TYPE diff;

  for(i=0; i<SIZE; i++) {
    diff = data->sol[i] - ref->sol[i];
    has_errors |= (diff<-EPSILON) || (EPSILON<diff);
  }

  // Return true if it's correct.
  return !has_errors;
}
