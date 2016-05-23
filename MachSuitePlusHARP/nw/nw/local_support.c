#include "nw.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  static unsigned *seqA_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->seqA), 1024); 
  static unsigned *seqB_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->seqB), 1024); 
  static unsigned *alignedA_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->alignedA), 1024); 
  static unsigned *alignedB_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->alignedB), 1024); 

  // Write our data set into device buffers  
  //
  memcpy(seqA_buffer, args->seqA, sizeof(args->seqA));
  memcpy(seqB_buffer, args->seqB, sizeof(args->seqB));
    
  // Set the arguments to our compute kernel
  //
  int status;
  status = clSetKernelArgSVMPointerAltera(kernel, 0, (void*)seqA_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 1, (void*)seqB_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 2, (void*)alignedA_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 3, (void*)alignedB_buffer);
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
  memcpy(args->alignedA, alignedA_buffer, sizeof(args->alignedA));
  memcpy(args->alignedB, alignedB_buffer, sizeof(args->alignedB));
}

/* Input format:
%% Section 1
char[]: sequence A
%% Section 2
char[]: sequence B
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_string(s, data->seqA, ALEN);

  s = find_section_start(p,2);
  parse_string(s, data->seqB, BLEN);

}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_string(fd, data->seqA, ALEN);

  write_section_header(fd);
  write_string(fd, data->seqB, BLEN);

  write_section_header(fd);
}

/* Output format:
%% Section 1
char[sum_size]: aligned sequence A
%% Section 2
char[sum_size]: aligned sequence B
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_string(s, data->alignedA, ALEN+BLEN);

  s = find_section_start(p,2);
  parse_string(s, data->alignedB, ALEN+BLEN);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_string(fd, data->alignedA, ALEN+BLEN);

  write_section_header(fd);
  write_string(fd, data->alignedB, ALEN+BLEN);

  write_section_header(fd);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

  has_errors |= memcmp(data->alignedA, ref->alignedA, ALEN+BLEN);
  has_errors |= memcmp(data->alignedB, ref->alignedB, ALEN+BLEN);

  // Return true if it's correct.
  return !has_errors;
}
