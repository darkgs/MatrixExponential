
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "common.h"
#include "matexp_matrix.h"

#include "matexp_gpu.h"

//#include "matexp.h"

static double rand_f(double s, double e) {
	if (s > e) {
		double t = s;
		s = e;
		e = t;
	}
	return s + ((double)rand()/(double)RAND_MAX) * (e-s);
}

static void rand_n0(double* n0, int dim) {
	for (int i=0; i<dim; ++i) {
		n0[i] = rand_f(-1., 1.);
	}
}

int exec_() {
	matexp_matrix* mat_input = matexp_create_matrix_from_file("data/DeplInfo_toy_BOC.txt");
	double *y = (double*)malloc(sizeof(double)*mat_input->dim);

	rand_n0(y, mat_input->dim);
	matexp_gpu_calc(mat_input, y, 4.7);

	matexp_release_matrix(mat_input);	mat_input = NULL;
	free(y);
	//matexp_finalize(me); me = NULL;
	return 0;
}

/*
int main(int argc, char *argv[]) {

	exec_();

	return 0;
	int opt;

	srand((unsigned int)time(NULL));

	matexp_matrix* mat_input = NULL;
	while((opt = getopt(argc, argv, "hf:")) != -1) {
		switch(opt) {
			case 'h':
				printf("usage: \n");
				break;
			case 'f':
				printf("input file: %s\n", optarg);
				mat_input = matexp_create_matrix_from_file(optarg);
				break;
		}
	}
	//matexp* me = matexp_init();	assert(me);
	
	double *y = (double*)malloc(sizeof(double)*mat_input->dim);
	rand_n0(y, mat_input->dim);
	matexp_gpu_calc(mat_input, y, 4.7);

	matexp_release_matrix(mat_input);	mat_input = NULL;
	free(y);
	//matexp_finalize(me); me = NULL;
	return 0;

	matexp_matrix* mat_bro = matexp_copy_matrix(mat_input);

	int matrix_dim = 16;
	func_ret_t ret;
	float *m;
	matexp_matrix** mm;

	int num_test = 1;

	mm = (matexp_matrix**)malloc(sizeof(matexp_matrix*)*num_test);
	for (int i=0; i<num_test; ++i) {
		mm[i] = matexp_create_matrix(matrix_dim);

		for (int c=0; c<matrix_dim; ++c) {
			for (int r=0; r<matrix_dim; ++r) {
				mm[i]->data[c*matrix_dim+r] = (c*matrix_dim+r) * 1.0f;
			}
		}
	}

	matexp_calc(me, mm, num_test);

	// release memory
	for (int i=0; i<num_test; ++i) {
		matexp_release_matrix(mm[i]); mm[i] = NULL;
	}
	free(mm);
	matexp_finalize(me); me = NULL;

	return 0;
}
*/
