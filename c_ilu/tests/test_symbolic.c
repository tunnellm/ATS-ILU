#include "ilu.h"

#include <stdio.h>
#include <string.h>

static CscMatrix *make_symbolic_regression_matrix(void)
{
    static const int edges[][2] = {
        {0, 9}, {1, 2}, {1, 6}, {2, 3}, {3, 7}, {3, 8},
        {4, 5}, {4, 7}, {5, 6}, {6, 9}, {7, 8},
    };
    enum { n = 10 };
    int adjacency[n][n];
    memset(adjacency, 0, sizeof(adjacency));
    for (int i = 0; i < n; ++i) {
        adjacency[i][i] = 1;
    }
    for (size_t e = 0; e < sizeof(edges) / sizeof(edges[0]); ++e) {
        const int i = edges[e][0];
        const int j = edges[e][1];
        adjacency[i][j] = 1;
        adjacency[j][i] = 1;
    }

    int nnz = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            nnz += adjacency[i][j];
        }
    }

    CscMatrix *A = csc_alloc(n, n, nnz);
    int p = 0;
    A->colptr[0] = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (adjacency[i][j]) {
                A->rowind[p] = i;
                A->x[p] = i == j ? 16.0 : -1.0;
                ++p;
            }
        }
        A->colptr[j + 1] = p;
    }
    return A;
}

static CscMatrix *make_unsymmetric_regression_matrix(void)
{
    static const int rowptr[] = {0, 1, 6, 13, 14, 15, 18, 22, 25, 28, 34, 35, 38};
    static const int columns[] = {
        0, 0, 1, 2, 6, 11, 0, 1, 2, 3, 5, 6, 9, 3, 4, 1, 5, 7, 1,
        5, 6, 8, 2, 4, 7, 5, 8, 9, 0, 4, 6, 8, 9, 10, 10, 5, 6, 11,
    };
    enum { n = 12, nnz = 38 };
    int col_counts[n];
    int next[n];
    memset(col_counts, 0, sizeof(col_counts));
    for (int p = 0; p < nnz; ++p) {
        ++col_counts[columns[p]];
    }

    CscMatrix *A = csc_alloc(n, n, nnz);
    A->colptr[0] = 0;
    for (int j = 0; j < n; ++j) {
        A->colptr[j + 1] = A->colptr[j] + col_counts[j];
        next[j] = A->colptr[j];
    }
    for (int i = 0; i < n; ++i) {
        for (int p = rowptr[i]; p < rowptr[i + 1]; ++p) {
            const int j = columns[p];
            const int q = next[j]++;
            A->rowind[q] = i;
            A->x[q] = i == j ? 16.0 : -1.0;
        }
    }
    return A;
}

static int same_pattern(const CscMatrix *A, const CscMatrix *B)
{
    return A->nrows == B->nrows && A->ncols == B->ncols && A->nnz == B->nnz &&
           memcmp(A->colptr, B->colptr,
                  ((size_t)A->ncols + 1u) * sizeof(int)) == 0 &&
           memcmp(A->rowind, B->rowind, (size_t)A->nnz * sizeof(int)) == 0;
}

static int check_mode(const CscMatrix *A, IluSymbolicMode mode, IluSymbolic *out)
{
    static const int expected_colptr[] = {0, 1, 2, 4, 6, 7, 9, 14, 19, 23, 27};
    static const int expected_rowind[] = {
        0, 1, 1, 2, 2, 3, 4, 4, 5, 1, 2, 3, 5, 6,
        3, 4, 5, 6, 7, 3, 6, 7, 8, 0, 6, 7, 9,
    };
    IluSymbolicTimings timings;
    if (ilu_build_symbolic_mode(A, 3, mode, out, &timings) != 0) {
        fprintf(stderr, "symbolic mode %d failed\n", mode);
        return -1;
    }
    if (!same_pattern(out->Lsym, out->Usym)) {
        fprintf(stderr,
                "symbolic mode %d broke symmetry: L_nnz=%d U_nnz=%d\n",
                mode, out->Lsym->nnz, out->Usym->nnz);
        return -1;
    }
    if (out->Usym->nnz != 27 ||
        memcmp(out->Usym->colptr, expected_colptr, sizeof(expected_colptr)) != 0 ||
        memcmp(out->Usym->rowind, expected_rowind, sizeof(expected_rowind)) != 0) {
        fprintf(stderr, "symbolic mode %d produced the wrong ILU(3) pattern\n", mode);
        return -1;
    }
    return 0;
}

static int check_unsymmetric_mode(const CscMatrix *A, IluSymbolicMode mode)
{
    static const int expected_l_colptr[] = {0, 1, 3, 6, 7, 8, 12, 17, 23, 27, 33, 34, 41};
    static const int expected_l_rowind[] = {
        0, 0, 1, 0, 1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 5, 6, 2, 3, 4, 5,
        6, 7, 5, 6, 7, 8, 0, 4, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10, 11,
    };
    static const int expected_u_colptr[] = {0, 1, 2, 4, 6, 7, 9, 13, 16, 19, 25, 27, 35};
    static const int expected_u_rowind[] = {
        0, 1, 1, 2, 2, 3, 4, 2, 5, 1, 2, 5, 6, 5, 6, 7, 6, 7,
        8, 2, 5, 6, 7, 8, 9, 9, 10, 1, 2, 5, 6, 7, 8, 9, 11,
    };
    IluSymbolic sym = {0};
    IluSymbolicTimings timings;
    int status = 0;

    if (ilu_build_symbolic_mode(A, 3, mode, &sym, &timings) != 0) {
        fprintf(stderr, "unsymmetric symbolic mode %d failed\n", mode);
        return -1;
    }
    if (sym.Lsym->nnz != 41 || sym.Usym->nnz != 35 ||
        memcmp(sym.Lsym->colptr, expected_l_colptr, sizeof(expected_l_colptr)) != 0 ||
        memcmp(sym.Lsym->rowind, expected_l_rowind, sizeof(expected_l_rowind)) != 0 ||
        memcmp(sym.Usym->colptr, expected_u_colptr, sizeof(expected_u_colptr)) != 0 ||
        memcmp(sym.Usym->rowind, expected_u_rowind, sizeof(expected_u_rowind)) != 0) {
        fprintf(stderr, "unsymmetric symbolic mode %d produced the wrong ILU(3) pattern\n", mode);
        status = -1;
    }
    ilu_symbolic_free(&sym);
    return status;
}

int main(void)
{
    CscMatrix *A = make_symbolic_regression_matrix();
    CscMatrix *A_unsymmetric = make_unsymmetric_regression_matrix();
    IluSymbolic serial = {0};
    IluSymbolic levelset = {0};
    int status = 0;

    if (check_mode(A, ILU_SYMBOLIC_SERIAL, &serial) != 0 ||
        check_mode(A, ILU_SYMBOLIC_LEVELSET, &levelset) != 0) {
        status = 1;
    } else if (!same_pattern(serial.Lsym, levelset.Lsym) ||
               !same_pattern(serial.Usym, levelset.Usym)) {
        fprintf(stderr, "serial and level-set symbolic patterns differ\n");
        status = 1;
    }
    if (check_unsymmetric_mode(A_unsymmetric, ILU_SYMBOLIC_SERIAL) != 0 ||
        check_unsymmetric_mode(A_unsymmetric, ILU_SYMBOLIC_LEVELSET) != 0) {
        status = 1;
    }

    ilu_symbolic_free(&levelset);
    ilu_symbolic_free(&serial);
    csc_free(A_unsymmetric);
    csc_free(A);
    if (status == 0) {
        puts("symbolic regression tests passed");
    }
    return status;
}
