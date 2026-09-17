#include "csc.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int row;
    int col;
    double val;
} CscEntry;

static int csc_entry_cmp(const void *a, const void *b)
{
    const CscEntry *ea = (const CscEntry *)a;
    const CscEntry *eb = (const CscEntry *)b;
    if (ea->col != eb->col) {
        return ea->col < eb->col ? -1 : 1;
    }
    if (ea->row != eb->row) {
        return ea->row < eb->row ? -1 : 1;
    }
    return 0;
}

static CscMatrix *csc_from_sorted_entries(int nrows, int ncols, int nnz, CscEntry *entries)
{
    qsort(entries, (size_t)nnz, sizeof(CscEntry), csc_entry_cmp);

    CscMatrix *B = csc_alloc(nrows, ncols, nnz);
    for (int p = 0; p < nnz; ++p) {
        B->colptr[entries[p].col + 1] += 1;
    }
    for (int j = 0; j < ncols; ++j) {
        B->colptr[j + 1] += B->colptr[j];
    }
    for (int p = 0; p < nnz; ++p) {
        B->rowind[p] = entries[p].row;
        B->x[p] = entries[p].val;
    }
    return B;
}

void *ilu_xmalloc(size_t nbytes)
{
    void *ptr = malloc(nbytes == 0 ? 1 : nbytes);
    if (!ptr) {
        fprintf(stderr, "out of memory while allocating %zu bytes\n", nbytes);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *ilu_xcalloc(size_t count, size_t size)
{
    void *ptr = calloc(count == 0 ? 1 : count, size == 0 ? 1 : size);
    if (!ptr) {
        fprintf(stderr, "out of memory while allocating %zu objects of %zu bytes\n", count, size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *ilu_xrealloc(void *ptr, size_t nbytes)
{
    void *newptr = realloc(ptr, nbytes == 0 ? 1 : nbytes);
    if (!newptr) {
        fprintf(stderr, "out of memory while reallocating %zu bytes\n", nbytes);
        exit(EXIT_FAILURE);
    }
    return newptr;
}

CscMatrix *csc_alloc(int nrows, int ncols, int nnz)
{
    CscMatrix *A = (CscMatrix *)ilu_xcalloc(1, sizeof(CscMatrix));
    A->nrows = nrows;
    A->ncols = ncols;
    A->nnz = nnz;
    A->colptr = (int *)ilu_xcalloc((size_t)ncols + 1, sizeof(int));
    A->rowind = (int *)ilu_xmalloc((size_t)nnz * sizeof(int));
    A->x = (double *)ilu_xmalloc((size_t)nnz * sizeof(double));
    return A;
}

CscMatrix *csc_clone(const CscMatrix *A)
{
    CscMatrix *B = csc_alloc(A->nrows, A->ncols, A->nnz);
    memcpy(B->colptr, A->colptr, ((size_t)A->ncols + 1) * sizeof(int));
    memcpy(B->rowind, A->rowind, (size_t)A->nnz * sizeof(int));
    memcpy(B->x, A->x, (size_t)A->nnz * sizeof(double));
    return B;
}

void csc_free(CscMatrix *A)
{
    if (!A) {
        return;
    }
    free(A->colptr);
    free(A->rowind);
    free(A->x);
    free(A);
}

void csc_set_all(CscMatrix *A, double value)
{
    for (int p = 0; p < A->nnz; ++p) {
        A->x[p] = value;
    }
}

void csc_zero(CscMatrix *A)
{
    memset(A->x, 0, (size_t)A->nnz * sizeof(double));
}

int csc_validate_sorted(const CscMatrix *A)
{
    if (!A || !A->colptr || !A->rowind || !A->x) {
        return 0;
    }
    if (A->colptr[0] != 0 || A->colptr[A->ncols] != A->nnz) {
        return 0;
    }
    for (int j = 0; j < A->ncols; ++j) {
        if (A->colptr[j] > A->colptr[j + 1]) {
            return 0;
        }
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            if (A->rowind[p] < 0 || A->rowind[p] >= A->nrows) {
                return 0;
            }
            if (p + 1 < A->colptr[j + 1] && A->rowind[p] >= A->rowind[p + 1]) {
                return 0;
            }
        }
    }
    return 1;
}

CscMatrix *csc_transpose_keepzeros(const CscMatrix *A)
{
    CscMatrix *AT = csc_alloc(A->ncols, A->nrows, A->nnz);
    int *counts = (int *)ilu_xcalloc((size_t)A->nrows, sizeof(int));

    for (int p = 0; p < A->nnz; ++p) {
        counts[A->rowind[p]] += 1;
    }

    AT->colptr[0] = 0;
    for (int j = 0; j < A->nrows; ++j) {
        AT->colptr[j + 1] = AT->colptr[j] + counts[j];
    }

    int *next = (int *)ilu_xmalloc((size_t)A->nrows * sizeof(int));
    memcpy(next, AT->colptr, (size_t)A->nrows * sizeof(int));

    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            const int i = A->rowind[p];
            const int q = next[i]++;
            AT->rowind[q] = j;
            AT->x[q] = A->x[p];
        }
    }

    free(next);
    free(counts);
    return AT;
}

CscMatrix *csc_triu_strict(const CscMatrix *A)
{
    int nnz = 0;
    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            nnz += A->rowind[p] < j;
        }
    }

    CscMatrix *U = csc_alloc(A->nrows, A->ncols, nnz);
    U->colptr[0] = 0;
    int q = 0;
    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            if (A->rowind[p] < j) {
                U->rowind[q] = A->rowind[p];
                U->x[q] = A->x[p];
                q += 1;
            }
        }
        U->colptr[j + 1] = q;
    }
    return U;
}

CscMatrix *csc_union_pattern(const CscMatrix *A, const CscMatrix *B)
{
    if (A->nrows != B->nrows || A->ncols != B->ncols) {
        return NULL;
    }

    int nnz = 0;
    for (int j = 0; j < A->ncols; ++j) {
        int pa = A->colptr[j];
        int pb = B->colptr[j];
        const int aend = A->colptr[j + 1];
        const int bend = B->colptr[j + 1];
        while (pa < aend || pb < bend) {
            int take;
            if (pb >= bend || (pa < aend && A->rowind[pa] < B->rowind[pb])) {
                take = A->rowind[pa++];
            } else if (pa >= aend || B->rowind[pb] < A->rowind[pa]) {
                take = B->rowind[pb++];
            } else {
                take = A->rowind[pa];
                pa += 1;
                pb += 1;
            }
            (void)take;
            nnz += 1;
        }
    }

    CscMatrix *S = csc_alloc(A->nrows, A->ncols, nnz);
    S->colptr[0] = 0;
    int q = 0;
    for (int j = 0; j < A->ncols; ++j) {
        int pa = A->colptr[j];
        int pb = B->colptr[j];
        const int aend = A->colptr[j + 1];
        const int bend = B->colptr[j + 1];
        while (pa < aend || pb < bend) {
            int row;
            if (pb >= bend || (pa < aend && A->rowind[pa] < B->rowind[pb])) {
                row = A->rowind[pa++];
            } else if (pa >= aend || B->rowind[pb] < A->rowind[pa]) {
                row = B->rowind[pb++];
            } else {
                row = A->rowind[pa];
                pa += 1;
                pb += 1;
            }
            S->rowind[q] = row;
            S->x[q] = 0.0;
            q += 1;
        }
        S->colptr[j + 1] = q;
    }
    return S;
}

CscMatrix *csc_permute_rows(const CscMatrix *A, const int *old_to_new)
{
    CscEntry *entries = (CscEntry *)ilu_xmalloc((size_t)A->nnz * sizeof(CscEntry));
    int q = 0;
    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            entries[q++] = (CscEntry){
                .row = old_to_new[A->rowind[p]],
                .col = j,
                .val = A->x[p],
            };
        }
    }
    CscMatrix *B = csc_from_sorted_entries(A->nrows, A->ncols, A->nnz, entries);
    free(entries);
    return B;
}

CscMatrix *csc_permute_rows_scale(
    const CscMatrix *A,
    const int *old_to_new,
    const double *row_log_scale,
    const double *col_log_scale)
{
    CscEntry *entries = (CscEntry *)ilu_xmalloc((size_t)A->nnz * sizeof(CscEntry));
    int q = 0;
    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            const int old_row = A->rowind[p];
            const double scale = exp(row_log_scale[old_row] + col_log_scale[j]);
            entries[q++] = (CscEntry){
                .row = old_to_new[old_row],
                .col = j,
                .val = A->x[p] * scale,
            };
        }
    }
    CscMatrix *B = csc_from_sorted_entries(A->nrows, A->ncols, A->nnz, entries);
    free(entries);
    return B;
}

CscMatrix *csc_permute_symmetric(const CscMatrix *A, const int *old_to_new)
{
    CscEntry *entries = (CscEntry *)ilu_xmalloc((size_t)A->nnz * sizeof(CscEntry));
    int q = 0;
    for (int j = 0; j < A->ncols; ++j) {
        const int new_col = old_to_new[j];
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            entries[q++] = (CscEntry){
                .row = old_to_new[A->rowind[p]],
                .col = new_col,
                .val = A->x[p],
            };
        }
    }
    CscMatrix *B = csc_from_sorted_entries(A->nrows, A->ncols, A->nnz, entries);
    free(entries);
    return B;
}

double csc_diag_value(const CscMatrix *A, int j)
{
    int lo = A->colptr[j];
    int hi = A->colptr[j + 1] - 1;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        const int row = A->rowind[mid];
        if (row == j) {
            return A->x[mid];
        }
        if (row < j) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return 0.0;
}

double csc_abs_checksum(const CscMatrix *A)
{
    double s = 0.0;
    for (int p = 0; p < A->nnz; ++p) {
        s += fabs(A->x[p]);
    }
    return s;
}

CscMatrix *csc_make_tridiagonal(int n)
{
    const int nnz = n <= 1 ? 1 : 3 * n - 2;
    CscMatrix *A = csc_alloc(n, n, nnz);
    int p = 0;
    A->colptr[0] = 0;
    for (int j = 0; j < n; ++j) {
        if (j > 0) {
            A->rowind[p] = j - 1;
            A->x[p++] = -1.0;
        }
        A->rowind[p] = j;
        A->x[p++] = 4.0;
        if (j + 1 < n) {
            A->rowind[p] = j + 1;
            A->x[p++] = -1.0;
        }
        A->colptr[j + 1] = p;
    }
    return A;
}

double wall_seconds(void)
{
    return omp_get_wtime();
}
