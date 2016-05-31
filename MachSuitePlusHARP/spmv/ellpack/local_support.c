#include "spmv.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON 1.0e-5

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  static unsigned *nzval_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->nzval), 1024); 
  static unsigned *cols_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->cols), 1024); 
  static unsigned *vec_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->vec), 1024); 
  static unsigned *out_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->out), 1024); 

  // Write our data set into device buffers  
  //
  memcpy(nzval_buffer, args->nzval, sizeof(args->nzval));
  memcpy(cols_buffer, args->cols, sizeof(args->cols));
  memcpy(vec_buffer, args->vec, sizeof(args->vec));
    
  // Set the arguments to our compute kernel
  //
  int status;
  status = clSetKernelArgSVMPointerAltera(kernel, 0, (void*)nzval_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 1, (void*)cols_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 2, (void*)vec_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 3, (void*)out_buffer);
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
  memcpy(args->out, out_buffer, sizeof(args->out));
}

/* Input format:
%% Section 1
TYPE[N*L]: the nonzeros of the matrix
%% Section 2
int32_t[N*L]: the column index of the nonzeros
%% Section 3
TYPE[N]: the dense vector
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->nzval, N*L);

  s = find_section_start(p,2);
  parse_int32_t_array(s, data->cols, N*L);

  s = find_section_start(p,3);
  STAC(parse_,TYPE,_array)(s, data->vec, N);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->nzval, N*L);

  write_section_header(fd);
  write_int32_t_array(fd, data->cols, N*L);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->vec, N);
}

/* Output format:
%% Section 1
TYPE[N]: The output vector
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->out, N);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->out, N);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i;
  TYPE diff;

  for(i=0; i<N; i++) {
    diff = data->out[i] - ref->out[i];
    has_errors |= (diff<-EPSILON) || (EPSILON<diff);
  }

  // Return true if it's correct.
  return !has_errors;
}
