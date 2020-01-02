
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <assert.h>

#include <complex.h>

#include <mkl.h>

#include "common.h"

//#include "matexp.h"
#include "matexp_gpu.h"

#ifdef _DEBUG_
static double rand_d(double s, double e) {
	if (s > e) {
		double t = s;
		s = e;
		e = t;
	}

	return s + ((double)rand()/(double)RAND_MAX) * (e-s);
}

static MKL_Complex8 rand_c(double s, double e) {
	MKL_Complex8 val;
	val.real = rand_d(s, e);
	val.imag = rand_d(s,e);

	return val;
}

static void mat_init_rand(MKL_Complex8 *mat, int C, int R) {

	for (int c=0; c<C; ++c) {
		for (int r=0; r<R; ++r) {
			mat[r*R+c] = rand_c(-1.0, 1.0);
		}
	}
}

static void mat_init_zero(MKL_Complex8 *mat, int C, int R) {
	for (int c=0; c<C; ++c) {
		for (int r=0; r<R; ++r) {
			mat[r*R+c].real = 0.0;
			mat[r*R+c].imag = 0.0;
		}
	}
}

static void mkl_mat_print(MKL_Complex8 *mat, int C, int R) {

	for (int r=0; r<R; ++r) {
		for (int c=0; c<C; ++c) {
			MKL_Complex8 val = mat[r*C+c];
			printf("(%.2f, %.2fI) ", val.real, val.imag);
		}
		printf("\n");
	}
}
#endif

/*
static void mkl_example() {
	int n = 6;
	MKL_Complex8 alpha, beta;
	alpha.real = 1.0;	alpha.imag = 0.0;
	beta.real = 0.0;	beta.imag = 0.0;

	MKL_Complex8 *A = (MKL_Complex8*)mkl_malloc(sizeof(MKL_Complex8)*n*n, 64);
	MKL_Complex8 *A_ = (MKL_Complex8*)mkl_malloc(sizeof(MKL_Complex8)*n*n, 64);
	MKL_Complex8 *Ret = (MKL_Complex8*)mkl_malloc(sizeof(MKL_Complex8)*n*n, 64);
	mat_init_rand(A, n, n);
	mat_init_zero(Ret, n, n);

	memcpy(A_, A, sizeof(MKL_Complex8)*n*n);

	lapack_int *p = (lapack_int*)mkl_malloc(sizeof(lapack_int)*n, 64);
	LAPACKE_cgetrf(LAPACK_ROW_MAJOR, n, n, A, n, p);
	LAPACKE_cgetri(LAPACK_ROW_MAJOR, n, A, n, p);

	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &alpha, A_, n, A, n, &beta, Ret, n);

	mkl_mat_print(Ret, n, n);
}
*/

static void mkl_inv_matrix(MKL_Complex8* mat, int dim) {
	MKL_Complex8 alpha, beta;
	alpha.real = 1.0;	alpha.imag = 0.0;
	beta.real = 0.0;	beta.imag = 0.0;

#ifdef _DEBUG_
	MKL_Complex8 *mat_ = (MKL_Complex8*)mkl_malloc(sizeof(MKL_Complex8)*dim*dim, 64);
	memcpy(mat_, mat, sizeof(MKL_Complex8)*dim*dim);

	MKL_Complex8 *ret_ = (MKL_Complex8*)mkl_malloc(sizeof(MKL_Complex8)*dim*dim, 64);
	mat_init_zero(ret_, dim, dim);
#endif

	lapack_int *p = (lapack_int*)mkl_malloc(sizeof(lapack_int)*dim, 64);
	LAPACKE_cgetrf(LAPACK_ROW_MAJOR, dim, dim, mat, dim, p);
	LAPACKE_cgetri(LAPACK_ROW_MAJOR, dim, mat, dim, p);

#ifdef _DEBUG_
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim,
			&alpha, mat_, dim, mat, dim, &beta, ret_, dim);
	printf("====== [DEBUG] should be I =========\n");
	mkl_mat_print(ret_, dim, dim);
#endif

#ifdef _DEBUG_
	mkl_free(mat_);
	mkl_free(ret_);
#endif
	mkl_free(p);
}

static double complex alpha_8[8] = {
	5.464930576870210e+3 - 3.797983575308356e+4 * I,
	9.045112476907548e+1 - 1.115537522430261e+3 * I,
	2.344818070467641e+2 - 4.228020157070496e+2 * I,
	9.453304067358312e+1 - 2.951294291446048e+2 * I,
	7.283792954673409e+2 - 1.205646080220011e+5 * I,
	3.648229059594851e+1 - 1.155509621409682e+2 * I,
	2.547321630156819e+1 - 2.639500283021502e+1 * I,
	2.394538338734709e+1 - 5.650522971778156e+0 * I,
};

static float theta_8[8][2] = {
	{3.509103608414918, 8.436198985884374},
	{5.948152268951177, 3.587457362018322},
	{5.264971343442647, 16.22022147316793},
	{1.419375897185666, 10.92536348449672},
	{6.416177699099435, 1.194122393370139},
	{4.993174737717997, 5.996881713603942},
	{-1.413928462488886, 13.49772569889275},
	{-10.84391707869699, 19.27744616718165},
};

/*
static void print_c_vector(double complex *v, int dim) {
	for (int i=0; i<dim; ++i) {
		printf("(%.3f + %.3fI) ", v[i]);
	}
	printf("\n");
}
*/

void matexp_gpu_calc(const matexp_matrix* mat, double *y, double dt) {
	int k = 8;
	double complex *alpha = alpha_8;
	static MKL_Complex8 *theta = NULL;
	if (!theta) {
		theta = (MKL_Complex8*)malloc(sizeof(MKL_Complex8)*k);
		for (int i=0; i<k; ++i) {
			theta[i].real = theta_8[i][0];
			theta[i].imag = theta_8[i][1];
		}
	}


	int dim = mat->dim;
	int nnz = mat->nnz;

	double complex *y_ = (double complex*)malloc(sizeof(double complex)*dim);

	// Row major dense matrix
	MKL_Complex8* A = (MKL_Complex8*)mkl_malloc(sizeof(MKL_Complex8)*dim*dim, 64);
	for (int l=0; l<k; ++l) {
		// init A to zero
		for (int r=0; r<dim; ++r) {
			for (int c=0; c<dim; ++c) {
				A[r*dim+c].real = 0.0f;
				A[r*dim+c].imag = 0.0f;
			}
		}
		// At -> A
		for (int i=0; i<nnz; ++i) {
			int r = mat->coo_row[i];
			int c = mat->coo_col[i];

			A[r*dim + c].real = mat->coo_val[i] * dt;
		}

		// At - I*theta -> A
		for (int i=0; i<dim; ++i) {
			int r = i, c = i;

			A[r*dim + c].real -= theta[l].real;
			A[r*dim + c].imag -= theta[l].imag;
		}
		
		// inverse of (At - I*theta) -> A
		mkl_inv_matrix(A, dim);

		// Ay -> y_
		for (int r=0; r<dim; ++r) {
			y_[r] = 0.0f + 0.0f*I;
			for (int c=0; c<dim; ++c) {
				double complex val = A[r*dim + c].real + A[r*dim + c].imag * I;
				y_[r] += val * y[c];
			}
		}

		// 2*real(alpha*y_) + [1,...,1] -> y
		for (int i=0; i<dim; ++i) {
			y[i] = 2.0f * creal(alpha[l]*y_[i]) + 1.0f;
		}
	}

	// alpha_0 * y -> y
	double alpha_0 = 2.124853710495224e-16;
	for (int i=0; i<dim; ++i) {
		y[i] *= alpha_0;
	}

	free(y_);
	mkl_free(A);
}

