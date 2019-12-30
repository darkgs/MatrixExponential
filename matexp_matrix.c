
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <assert.h>

#include "common.h"
#include "matexp_matrix.h"

/*
static void convert_dense2csc(matexp_matrix* mat) {
	assert(mat->dim > 0 && mat->data);

	int dim = mat->dim;

	// count nnz
	int nnz = 0;
	for (int r=0; r<dim; ++r) {
		for (int c=0; c<dim; ++c) {
			if (mat->data[c*dim + r] == 0.0)
				continue;
			nnz++;
		}
	}
	printf("nnz: %d\n", nnz);

	//
	if (mat->csc_values)	free(mat->csc_values);
	if (mat->csc_rows)		free(mat->csc_rows);
	if (mat->csc_col_ptrs)	free(mat->csc_col_ptrs);

	mat->csc_values = (double*)malloc(sizeof(double)*nnz);
	mat->csc_rows = (int*)malloc(sizeof(int)*nnz);
	mat->csc_col_ptrs = (int*)malloc(sizeof(int)*(dim+1));

	int cur_val = 0;
	for (int c=0; c<dim; ++c) {
		mat->csc_col_ptrs[c] = cur_val;
		for (int r=0; r<dim; ++r) {
			if (mat->data[c*dim + r] == 0.0)
				continue;

			mat->csc_values[cur_val] = mat->data[c*dim + r];
			mat->csc_rows[cur_val] = r;

			cur_val++;
		}
	}
	assert(cur_val == nnz);
	mat->csc_col_ptrs[dim] = cur_val;
}
*/

matexp_matrix* matexp_alloc_matrix() {
    matexp_matrix* mat = (matexp_matrix*)malloc(sizeof(matexp_matrix));
    //mat->data = NULL;
    mat->dim = -1;
    //mat->lu = NULL;
	mat->nnz = -1;
	
	// saved as COO
	mat->coo_col = NULL;
	mat->coo_row = NULL;
	mat->coo_val = NULL;

	/*
	mat->csc_values = NULL;
	mat->csc_rows = NULL;
	mat->csc_col_ptrs = NULL;
	*/

	return mat;
}

void matexp_release_matrix(matexp_matrix* mat) {
    //if (mat->data)  free(mat->data);
    //if (mat->lu)    free(mat->lu);
	if (mat->coo_col)	free(mat->coo_col);
	if (mat->coo_row)	free(mat->coo_row);
	if (mat->coo_val)	free(mat->coo_val);

    free(mat);
}

/*
matexp_matrix* matexp_create_matrix(size_t matrix_dim) {
    func_ret_t ret;

    matexp_matrix* mat = matexp_alloc_matrix();

    ret = create_matrix(&mat->data, matrix_dim);
    if (ret != RET_SUCCESS) {
        matexp_release_matrix(mat);
        printf("Failed to create matrix internally size=%d\n", matrix_dim);
        return NULL;
    }
    mat->dim = matrix_dim;

    return mat;
}
*/

matexp_matrix* matexp_copy_matrix(const matexp_matrix* mat_ori) {
    matexp_matrix* mat = matexp_alloc_matrix();

	if (mat_ori->dim < 0)
		return mat;

	int dim = mat_ori->dim;
	int nnz = mat_ori->nnz;

	mat->dim = dim;
	mat->nnz = nnz;

	if (mat->coo_col) {
		mat->coo_col = (int*)malloc(sizeof(int)*nnz);
		memcpy(mat->coo_col, mat_ori->coo_col, sizeof(int)*nnz);
	}
	if (mat->coo_row) {
		mat->coo_row = (int*)malloc(sizeof(int)*nnz);
		memcpy(mat->coo_row, mat_ori->coo_row, sizeof(int)*nnz);
	}
	if (mat->coo_val) {
		mat->coo_val = (double*)malloc(sizeof(double)*nnz);
		memcpy(mat->coo_val, mat_ori->coo_val, sizeof(double)*nnz);
	}

	/*
	if (mat_ori->data) {
		mat->data = (double*)malloc(sizeof(double)*dim*dim);
		memcpy(mat->data, mat_ori->data, sizeof(double)*dim*dim);
	}

	if (mat_ori->lu) {
		mat->lu = (double*)malloc(sizeof(double)*dim*dim);
		memcpy(mat->lu, mat_ori->lu, sizeof(double)*dim*dim);
	}
	*/

	return mat;
}

void matexp_matrix_print(const matexp_matrix* mat) {
	for (int i=0; i<mat->nnz; ++i) {
		printf("(%d, %d, %.4f)\n", mat->coo_row[i], mat->coo_col[i], mat->coo_val[i]);
	}
	printf("dim: %d\n", mat->dim);
}

matexp_matrix* matexp_create_matrix_from_file(const char* file_path) {

    matexp_matrix* mat = matexp_alloc_matrix();
	
	int dim;
	FILE *fp = NULL;
	fp = fopen(file_path, "rb");
	assert(fp != NULL);

	fscanf(fp, "%d\n", &dim);

	int* coo_col = (int*)malloc(sizeof(int)*dim*dim);
	int* coo_row = (int*)malloc(sizeof(int)*dim*dim);
	double* coo_val = (double*)malloc(sizeof(double)*dim*dim);
	double val;
	int nnz = 0;
	for (int r=0; r<dim; ++r) {
		for (int c=0; c<dim; ++c) {
			fscanf(fp, "%E ", &val);
			if (val == 0.0f)
				continue;
			coo_row[nnz] = r;
			coo_col[nnz] = c;
			coo_val[nnz] = val;
			nnz++;
		}
	}

	mat->dim = dim;
	mat->nnz = nnz;

	mat->coo_col = (int*)malloc(sizeof(int)*nnz);
	mat->coo_row = (int*)malloc(sizeof(int)*nnz);
	mat->coo_val = (double*)malloc(sizeof(double)*nnz);

	memcpy(mat->coo_col, coo_col, sizeof(int)*nnz);
	memcpy(mat->coo_row, coo_row, sizeof(int)*nnz);
	memcpy(mat->coo_val, coo_val, sizeof(double)*nnz);

	free(coo_col);
	free(coo_row);
	free(coo_val);
	
	//convert_dense2csc(mat);

	return mat;
}

/*
void matexp_matrix_lu_alloc(matexp_matrix* mat) {
    if (mat->lu)
        return;
    mat->lu = (double*)malloc(sizeof(double)*mat->dim*mat->dim);
}

void matexp_matrix_lu_free(matexp_matrix* mat) {
    if (!mat->lu)
        return;

    free(mat->lu); mat->lu = NULL;
}
*/


