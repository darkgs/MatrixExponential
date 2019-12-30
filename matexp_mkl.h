#ifndef _MATEXP_MKL_H
#define _MATEXP_MKL_H

#include "matexp_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

void matexp_mkl_calc(const matexp_matrix* mat, double *y, double dt);

#ifdef __cplusplus
}
#endif

#endif // _MATEXP_MKL_H
