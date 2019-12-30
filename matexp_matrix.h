
#ifndef _MATEXP_MATRIX_H
#define _MATEXP_MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _matexp_matrix {
    size_t dim;
    //float* data;
    //float* lu;
	int nnz;

	// COO format
	int* coo_col;
	int* coo_row;
	double* coo_val;

	// CSC format
	/*
	float* csc_values;
	int* csc_rows;
	int* csc_col_ptrs;
	*/
} matexp_matrix;

matexp_matrix* matexp_alloc_matrix();
void matexp_release_matrix(matexp_matrix* mat);

//matexp_matrix* matexp_create_matrix(size_t matrix_dim);
matexp_matrix* matexp_create_matrix_from_file(const char* file_path);

/*
void matexp_matrix_lu_alloc(matexp_matrix* mat);
void matexp_matrix_lu_free(matexp_matrix* mat);
void matexp_matrix_lu_valid(matexp_matrix* mat);
*/

matexp_matrix* matexp_copy_matrix(const matexp_matrix* mat_ori);
void matexp_matrix_print(const matexp_matrix* mat);

#ifdef __cplusplus
}
#endif

#endif  //  _MATEXP_MATRIX_H
