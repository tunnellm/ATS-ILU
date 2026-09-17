#ifndef C_ILU_CSC_H
#define C_ILU_CSC_H

#include <stddef.h>

typedef struct {
    int nrows;
    int ncols;
    int nnz;
    int *colptr;
    int *rowind;
    double *x;
} CscMatrix;

void *ilu_xmalloc(size_t nbytes);
void *ilu_xcalloc(size_t count, size_t size);
void *ilu_xrealloc(void *ptr, size_t nbytes);

CscMatrix *csc_alloc(int nrows, int ncols, int nnz);
CscMatrix *csc_clone(const CscMatrix *A);
void csc_free(CscMatrix *A);
void csc_set_all(CscMatrix *A, double value);
void csc_zero(CscMatrix *A);
int csc_validate_sorted(const CscMatrix *A);

CscMatrix *csc_transpose_keepzeros(const CscMatrix *A);
CscMatrix *csc_triu_strict(const CscMatrix *A);
CscMatrix *csc_union_pattern(const CscMatrix *A, const CscMatrix *B);
CscMatrix *csc_permute_rows(const CscMatrix *A, const int *old_to_new);
CscMatrix *csc_permute_rows_scale(
    const CscMatrix *A,
    const int *old_to_new,
    const double *row_log_scale,
    const double *col_log_scale);
CscMatrix *csc_permute_symmetric(const CscMatrix *A, const int *old_to_new);
double csc_diag_value(const CscMatrix *A, int j);
double csc_abs_checksum(const CscMatrix *A);

CscMatrix *csc_read_matrix_market(const char *path);
CscMatrix *csc_make_tridiagonal(int n);

double wall_seconds(void);

#endif
