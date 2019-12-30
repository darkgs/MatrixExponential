
#include <stdarg.h>
#include <stdio.h>
#include <sys/time.h>

#include <assert.h>

#include <CL/cl.h>

#include "common.h"
#include "matexp.h"

static cl_mem matexp_lud(matexp* me, matexp_matrix* mat, cl_command_queue cmd_queue);
void matexp_calc(matexp* me, matexp_matrix** mats, size_t num_mat) {

    stopwatch sw, sw2;
    double reco_time = 0.0;

    stopwatch_start(&sw);
    cl_mem* cl_mem_to_be_free = (cl_mem*)malloc(sizeof(cl_mem)*num_mat);

	size_t num_devices = MIN(me->num_devices, 1);
	size_t max_batch = 3000 * num_devices;

	matexp_log("calc - num_devices: %d\n", num_devices);

    int i, k, j, r, c, m;
    //
    for (i=0; i<num_mat; ++i) {
        matexp_matrix_lu_alloc(mats[i]);
    }

    //
	for (j=0; j<num_mat/max_batch+1; ++j) {
		int i_st = j*max_batch;
		int i_ed = MIN((j+1)*max_batch, num_mat);

    	for (i=i_st; i<i_ed; ++i) {
        	cl_mem_to_be_free[i] = matexp_lud(me, mats[i], me->cmd_queue[i%num_devices]);
    	}

    	for (k=0; k<me->num_devices; ++k) {
	    	clFinish(me->cmd_queue[k]);
    	}

    	for (i=i_st; i<i_ed; ++i) {
	    	clReleaseMemObject(cl_mem_to_be_free[i]);
    	}
	}
    stopwatch_stop(&sw);
    matexp_log("GPU calc (ms): %lf\n", 1000*get_interval_by_sec(&sw));
#ifdef _DEBUG
    for (i=0; i<num_mat; ++i) {
        matexp_matrix_lu_valid(mats[i]);
    }
#endif

    //
    for (i=0; i<num_mat; ++i) {
        matexp_matrix* mat = mats[i];
        matexp_matrix_lu_free(mat);
    }

    return 0;

    stopwatch_start(&sw);
    #pragma omp parallel for schedule(static)
    for (m=0; m<num_mat; ++m) {
        matexp_matrix* mat = mats[m];
        size_t matrix_dim = mat->dim;
        // inverse of L [TODO] check code
        for (i=matrix_dim-1; i>=0; i--) {
            mat->data[i*matrix_dim+i] = 1.0;

            for (r=matrix_dim-1; r>=0; r--) {
                for (c=0; c<matrix_dim; c++) {
                    mat->data[r*matrix_dim + c] -= \
                        mat->data[i*matrix_dim+c] * mat->data[r*matrix_dim+i];
                }
            }
        }

        // [TODO] inverse of R
        for (i=matrix_dim-1; i>=0; i--) {
            mat->data[i*matrix_dim+i] = 1.0;

            for (r=matrix_dim-1; r>=0; r--) {
                for (c=0; c<matrix_dim; c++) {
                    mat->data[r*matrix_dim + c] -= \
                        mat->data[i*matrix_dim+c] * mat->data[r*matrix_dim+i];
                }
            }
        }

        // reconstruct matrix
        float *tmp = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));
        float *lu = mat->data;
        for (i=0; i < matrix_dim; i ++) {
            for (j=0; j< matrix_dim; j++) {
                float sum = 0;
                float l,u;
                for (k=0; k <= MIN(i,j); k++){
                    l = (i==k) ? 1 : lu[i*matrix_dim+k];
                    u = lu[k*matrix_dim+j];
                    sum += l*u;
                }
                tmp[i*matrix_dim+j] = sum;
            }

        }
        mat->data = tmp;
        free(lu);
    }
    stopwatch_stop(&sw);
    matexp_log("CPU calc (ms): %lf\n", 1000*get_interval_by_sec(&sw));
}

void matexp_log(const char* format, ...) {
    char* log;
    va_list args;

    va_start(args, format);
    if(0 > vasprintf(&log, format, args)) log = NULL;
    va_end(args);

    if (!log) return;

    printf("[matexp] %s", log);
    free(log);

}

matexp* matexp_init() {
    int use_gpu = 1;
    size_t size = -1;
	cl_int result = 0, err = 0;

    matexp* me = (matexp*)malloc(sizeof(matexp));

    me->context = 0;
    me->cmd_queue = 0;
    me->device_type = 0;
    me->device_list = 0;
    me->num_devices = 0;

    me->diagnal = 0;
	me->perimeter = 0;
    me->internal = 0;

    // create OpenCL context
	cl_platform_id platform_id;
	if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) {
        matexp_finalize(me); printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return NULL;
    }

	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	me->device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

	me->context = clCreateContextFromType( ctxprop, me->device_type, NULL, NULL, NULL );
	if(!me->context) { 
        matexp_finalize(me); matexp_log("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU");
        return NULL;
    }

	// get the list of GPUs
	result = clGetContextInfo( me->context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	me->num_devices = (int) (size / sizeof(cl_device_id));

	if( result != CL_SUCCESS || me->num_devices < 1 ) {
        matexp_finalize(me); matexp_log("ERROR: clGetContextInfo() failed\n"); return NULL;
    }

	me->device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*me->num_devices);
	if( !me->device_list ) {
        matexp_finalize(me); matexp_log("ERROR: new cl_device_id[] failed\n"); return NULL;
    }
	result = clGetContextInfo( me->context, CL_CONTEXT_DEVICES, size, me->device_list, NULL );
	if( result != CL_SUCCESS ) {
        matexp_finalize(me); matexp_log("ERROR: clGetContextInfo() failed\n"); return NULL;
    }

	// create command queue for the first device
    me->cmd_queue = (cl_command_queue*)malloc(sizeof(cl_command_queue)*me->num_devices);
    for (int i=0; i<me->num_devices; ++i) {
	    me->cmd_queue[i] = clCreateCommandQueue( me->context, me->device_list[i], 0, NULL );
	    if( !me->cmd_queue[i] ) {
            matexp_finalize(me); printf("ERROR: clCreateCommandQueue() failed\n"); return NULL;
        }
    }

    // compile kernel codes
	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	if(!source) {
        matexp_finalize(me); printf("ERROR: calloc(%d) failed\n", sourcesize); return NULL;
    }

	char * kernel_lud_diag   = "lud_diagonal";
	char * kernel_lud_peri   = "lud_perimeter";
	char * kernel_lud_inter  = "lud_internal";
	FILE * fp = fopen("./lud_kernel.cl", "rb"); 
	if(!fp) {
        matexp_finalize(me); printf("ERROR: unable to open '%s'\n"); return NULL;
    }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(me->context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) {
        matexp_finalize(me); printf("ERROR: clCreateProgramWithSource() => %d\n", err); return NULL;
    }
	char clOptions[110];
	sprintf(clOptions," ");
#ifdef MATEXP_BLOCK_SIZE
	sprintf(clOptions + strlen(clOptions), " -DBLOCK_SIZE=%d", MATEXP_BLOCK_SIZE);
#endif

	err = clBuildProgram(prog, 0, NULL, clOptions, NULL, NULL);
	if(err != CL_SUCCESS) {
        matexp_finalize(me); printf("ERROR: clBuildProgram() => %d\n", err); return NULL;
    }

	me->diagnal   = clCreateKernel(prog, kernel_lud_diag, &err);  
	me->perimeter = clCreateKernel(prog, kernel_lud_peri, &err);  
	me->internal  = clCreateKernel(prog, kernel_lud_inter, &err);  

	clReleaseProgram(prog);

    return me;
}

void matexp_finalize(matexp* me) {
	// release OpenCL resouces
    if( me->diagnal ) clReleaseKernel( me->diagnal );
	if( me->perimeter ) clReleaseKernel( me->perimeter );
	if( me->internal ) clReleaseKernel( me->internal );

	if( me->cmd_queue ) {
        for (int i=0; i<me->num_devices; ++i) {
            clReleaseCommandQueue( me->cmd_queue[i] );
        }
    }
	if( me->context ) clReleaseContext( me->context );
	if( me->device_list ) free( me->device_list );

    free(me);
}

struct matexp_calc_args {
    matexp* me;
    matexp_matrix* mat;
};

static cl_mem matexp_lud(matexp* me, matexp_matrix* mat, cl_command_queue cmd_queue) {
    size_t matrix_dim = mat->dim;
    cl_context context = me->context;

	cl_mem d_m;
	cl_int err = 0;

	cl_kernel diagnal = me->diagnal;
    cl_kernel perimeter = me->perimeter;
    cl_kernel internal = me->internal;

    int i, j, k;

    assert(mat->lu);

    // LUD decomposition
	d_m = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_dim*matrix_dim * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) {
        matexp_log("ERROR: clCreateBuffer d_m (size:%d) => %d\n", matrix_dim*matrix_dim, err);
    } 

	err = clEnqueueWriteBuffer(cmd_queue, d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), mat->data, 0, 0, 0);
	if(err != CL_SUCCESS) {
        matexp_log("ERROR: clEnqueueWriteBuffer d_m (size:%d) => %d\n", matrix_dim*matrix_dim, err);
    }

	for (i=0; i < matrix_dim-MATEXP_BLOCK_SIZE; i += MATEXP_BLOCK_SIZE) {
        clSetKernelArg(diagnal, 0, sizeof(void *), (void*) &d_m);
        clSetKernelArg(diagnal, 1, sizeof(float) * MATEXP_BLOCK_SIZE * MATEXP_BLOCK_SIZE, (void*)NULL );
        clSetKernelArg(diagnal, 2, sizeof(cl_int), (void*) &matrix_dim);
        clSetKernelArg(diagnal, 3, sizeof(cl_int), (void*) &i);

        size_t global_work1[3]  = {MATEXP_BLOCK_SIZE, 1, 1};
        size_t local_work1[3]  = {MATEXP_BLOCK_SIZE, 1, 1};

        err = clEnqueueNDRangeKernel(cmd_queue, diagnal, 2, NULL, global_work1, local_work1, 0, 0, 0);
        if(err != CL_SUCCESS) {
            matexp_log("ERROR:  diagnal clEnqueueNDRangeKernel()=>%d failed\n", err);
        }

        clSetKernelArg(perimeter, 0, sizeof(void *), (void*) &d_m);
        clSetKernelArg(perimeter, 1, sizeof(float) * MATEXP_BLOCK_SIZE * MATEXP_BLOCK_SIZE, (void*)NULL );
        clSetKernelArg(perimeter, 2, sizeof(float) * MATEXP_BLOCK_SIZE * MATEXP_BLOCK_SIZE, (void*)NULL );
        clSetKernelArg(perimeter, 3, sizeof(float) * MATEXP_BLOCK_SIZE * MATEXP_BLOCK_SIZE, (void*)NULL );
        clSetKernelArg(perimeter, 4, sizeof(cl_int), (void*) &matrix_dim);
        clSetKernelArg(perimeter, 5, sizeof(cl_int), (void*) &i);

        size_t global_work2[3] = {MATEXP_BLOCK_SIZE * 2 * ((matrix_dim-i)/MATEXP_BLOCK_SIZE-1), 1, 1};
        size_t local_work2[3]  = {MATEXP_BLOCK_SIZE * 2, 1, 1};

        err = clEnqueueNDRangeKernel(cmd_queue, perimeter, 2, NULL, global_work2, local_work2, 0, 0, 0);
        if(err != CL_SUCCESS) {
            matexp_log("ERROR:  perimeter clEnqueueNDRangeKernel()=>%d failed\n", err);
        }

        clSetKernelArg(internal, 0, sizeof(void *), (void*) &d_m);
        clSetKernelArg(internal, 1, sizeof(float) * MATEXP_BLOCK_SIZE * MATEXP_BLOCK_SIZE, (void*)NULL );
        clSetKernelArg(internal, 2, sizeof(float) * MATEXP_BLOCK_SIZE * MATEXP_BLOCK_SIZE, (void*)NULL );
        clSetKernelArg(internal, 3, sizeof(cl_int), (void*) &matrix_dim);
        clSetKernelArg(internal, 4, sizeof(cl_int), (void*) &i);

        size_t global_work3[3] = {MATEXP_BLOCK_SIZE * ((matrix_dim-i)/MATEXP_BLOCK_SIZE-1),
            MATEXP_BLOCK_SIZE * ((matrix_dim-i)/MATEXP_BLOCK_SIZE-1), 1};
        size_t local_work3[3] = {MATEXP_BLOCK_SIZE, MATEXP_BLOCK_SIZE, 1};

        err = clEnqueueNDRangeKernel(cmd_queue, internal, 2, NULL, global_work3, local_work3, 0, 0, 0);
        if(err != CL_SUCCESS) {
            matexp_log("ERROR:  internal clEnqueueNDRangeKernel()=>%d failed\n", err);
        }	
    }

	clSetKernelArg(diagnal, 0, sizeof(void *), (void*) &d_m);
	clSetKernelArg(diagnal, 1, sizeof(float) * MATEXP_BLOCK_SIZE * MATEXP_BLOCK_SIZE, (void*)NULL );
	clSetKernelArg(diagnal, 2, sizeof(cl_int), (void*) &matrix_dim);
	clSetKernelArg(diagnal, 3, sizeof(cl_int), (void*) &i);

	size_t global_work1[3]  = {MATEXP_BLOCK_SIZE, 1, 1};
	size_t local_work1[3]  = {MATEXP_BLOCK_SIZE, 1, 1};
	err = clEnqueueNDRangeKernel(cmd_queue, diagnal, 2, NULL, global_work1, local_work1, 0, 0, 0);
	if(err != CL_SUCCESS) {
        matexp_log("ERROR:  diagnal clEnqueueNDRangeKernel()=>%d failed\n", err);
    }

	err = clEnqueueReadBuffer(cmd_queue, d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), mat->lu, 0, 0, 0);
	if(err != CL_SUCCESS) {
        matexp_log("ERROR: clEnqueueReadBuffer  d_m (size:%d) => %d\n", matrix_dim*matrix_dim, err);
    }

    return d_m;
}


