#ifndef _MATEXP_H
#define _MATEXP_H

#define MATEXP_BLOCK_SIZE 16

#include <CL/cl.h>

#include "matexp_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _matexp {
    cl_context context;
    cl_command_queue* cmd_queue;
    cl_device_type device_type;
    cl_device_id* device_list;
    cl_int num_devices;

	cl_kernel diagnal, perimeter, internal;

} matexp;

void matexp_log(const char* format, ...);

matexp* matexp_init();
void matexp_finalize(matexp* me);

void matexp_calc(matexp* me, matexp_matrix** mat, size_t num_mat);

#ifdef __cplusplus
}
#endif

#endif  //  _MATEXP_H
