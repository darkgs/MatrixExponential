#ifndef _MATEXP_GPU_H
#define _MATEXP_GPU_H

#include "matexp_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

void matexp_gpu_calc(const matexp_matrix* mat, double *y, double dt);

#ifdef __cplusplus
}
#endif

#endif // _MATEXP_GPU_H
