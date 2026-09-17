#include "preprocess.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef C_ILU_HAVE_MC64
#include "superlu_ddefs.h"

extern int mc64id_dist(int *icntl);
extern int mc64ad_dist(int *job, int *n, int_t *ne, int_t *ip,
                       int_t *irn, double *a, int *num, int_t *cperm,
                       int_t *liw, int_t *iw, int_t *ldw, double *dw,
                       int *icntl, int *info);
#endif

PreprocessMode preprocess_effective_mode(PreprocessMode mode)
{
#ifndef C_ILU_HAVE_MC64
    if (mode == PREPROCESS_MC64) return PREPROCESS_NONE;
    if (mode == PREPROCESS_MC64_RCM) return PREPROCESS_RCM;
#endif
    return mode;
}

void preprocess_warn_unavailable(void)
{
    fprintf(stderr,
        "\n********************************************************************\n"
        "WARNING: MC64 MATCHING AND SCALING WERE NOT APPLIED\n"
        "MC64 is not included in this package. No matching or MC64 scaling\n"
        "is performed; RCM, if requested, is still applied. These results\n"
        "DO NOT reproduce the paper's unsymmetric preprocessing.\n"
        "Supply external MC64 using MC64_SOURCE or MC64_LIBS and rebuild.\n"
        "********************************************************************\n\n");
}

typedef struct {
    int v;
    int degree;
} Neighbor;

static int neighbor_cmp(const void *a, const void *b)
{
    const Neighbor *na = (const Neighbor *)a;
    const Neighbor *nb = (const Neighbor *)b;
    if (na->degree != nb->degree) {
        return na->degree < nb->degree ? -1 : 1;
    }
    if (na->v != nb->v) {
        return na->v < nb->v ? -1 : 1;
    }
    return 0;
}

const char *preprocess_mode_name(PreprocessMode mode)
{
    switch (mode) {
    case PREPROCESS_NONE:
        return "none";
    case PREPROCESS_RCM:
        return "rcm";
    case PREPROCESS_MC64:
        return "mc64";
    case PREPROCESS_MC64_RCM:
        return "mc64-rcm";
    case PREPROCESS_DIAG:
        return "diag";
    case PREPROCESS_DIAG_RCM:
        return "diag-rcm";
    case PREPROCESS_LEFT_DIAG:
        return "left-diag";
    case PREPROCESS_LEFT_DIAG_RCM:
        return "left-diag-rcm";
    }
    return "unknown";
}

int parse_preprocess_mode(const char *name, PreprocessMode *mode)
{
    if (strcmp(name, "none") == 0) {
        *mode = PREPROCESS_NONE;
    } else if (strcmp(name, "rcm") == 0) {
        *mode = PREPROCESS_RCM;
    } else if (strcmp(name, "mc64") == 0) {
        *mode = PREPROCESS_MC64;
    } else if (strcmp(name, "mc64-rcm") == 0) {
        *mode = PREPROCESS_MC64_RCM;
    } else if (strcmp(name, "diag") == 0) {
        *mode = PREPROCESS_DIAG;
    } else if (strcmp(name, "diag-rcm") == 0) {
        *mode = PREPROCESS_DIAG_RCM;
    } else if (strcmp(name, "left-diag") == 0) {
        *mode = PREPROCESS_LEFT_DIAG;
    } else if (strcmp(name, "left-diag-rcm") == 0) {
        *mode = PREPROCESS_LEFT_DIAG_RCM;
    } else {
        return -1;
    }
    return 0;
}

static int preprocess_transform_init(PreprocessTransform *transform, int n)
{
    if (!transform) {
        return 0;
    }
    memset(transform, 0, sizeof(*transform));
    transform->n = n;
    transform->row_old_to_new = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    transform->col_old_to_new = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    transform->row_scale = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    transform->col_scale = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        transform->row_old_to_new[i] = i;
        transform->col_old_to_new[i] = i;
        transform->row_scale[i] = 1.0;
        transform->col_scale[i] = 1.0;
    }
    return 0;
}

void preprocess_transform_free(PreprocessTransform *transform)
{
    if (!transform) {
        return;
    }
    free(transform->row_old_to_new);
    free(transform->col_old_to_new);
    free(transform->row_scale);
    free(transform->col_scale);
    memset(transform, 0, sizeof(*transform));
}

static void transform_apply_symmetric_perm(PreprocessTransform *transform, const int *old_to_new)
{
    if (!transform) {
        return;
    }
    for (int i = 0; i < transform->n; ++i) {
        transform->row_old_to_new[i] = old_to_new[transform->row_old_to_new[i]];
        transform->col_old_to_new[i] = old_to_new[transform->col_old_to_new[i]];
    }
}

static void transform_apply_current_diag_scale(PreprocessTransform *transform, const double *d)
{
    if (!transform) {
        return;
    }
    for (int i = 0; i < transform->n; ++i) {
        transform->row_scale[i] *= d[transform->row_old_to_new[i]];
        transform->col_scale[i] *= d[transform->col_old_to_new[i]];
    }
}

static void transform_apply_current_left_diag_scale(
    PreprocessTransform *transform,
    const double *d)
{
    if (!transform) {
        return;
    }
    for (int i = 0; i < transform->n; ++i) {
        transform->row_scale[i] *= d[transform->row_old_to_new[i]];
    }
}

#ifdef C_ILU_HAVE_MC64
static void transform_apply_rowperm_log_scale(
    PreprocessTransform *transform,
    const int *row_old_to_new,
    const double *row_log_scale,
    const double *col_log_scale)
{
    if (!transform) {
        return;
    }
    for (int i = 0; i < transform->n; ++i) {
        const int current_row = transform->row_old_to_new[i];
        transform->row_old_to_new[i] = row_old_to_new[current_row];
        transform->row_scale[i] *= exp(row_log_scale[current_row]);
    }
    for (int j = 0; j < transform->n; ++j) {
        const int current_col = transform->col_old_to_new[j];
        transform->col_scale[j] *= exp(col_log_scale[current_col]);
    }
}

#endif

static void invert_permutation_old_to_new(const int *old_at_new, int *old_to_new, int n)
{
    for (int new_i = 0; new_i < n; ++new_i) {
        old_to_new[old_at_new[new_i]] = new_i;
    }
}

static CscMatrix *symmetrized_pattern(const CscMatrix *A)
{
    CscMatrix *AT = csc_transpose_keepzeros(A);
    CscMatrix *S = csc_union_pattern(A, AT);
    csc_free(AT);
    return S;
}

static int compute_rcm_old_to_new(const CscMatrix *A, int *old_to_new, int *levels_hint)
{
    const int n = A->ncols;
    CscMatrix *S = symmetrized_pattern(A);
    int *degree = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    int *visited = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    int *queue = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    int *order = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    Neighbor *neighbors = (Neighbor *)ilu_xmalloc((size_t)n * sizeof(Neighbor));
    int order_len = 0;
    int max_component_depth = 0;

    for (int j = 0; j < n; ++j) {
        int d = 0;
        for (int p = S->colptr[j]; p < S->colptr[j + 1]; ++p) {
            d += S->rowind[p] != j;
        }
        degree[j] = d;
    }

    while (order_len < n) {
        int start = -1;
        int best_degree = 0;
        for (int i = 0; i < n; ++i) {
            if (!visited[i] && (start < 0 || degree[i] < best_degree)) {
                start = i;
                best_degree = degree[i];
            }
        }
        if (start < 0) {
            break;
        }

        int head = 0;
        int tail = 0;
        int depth_tail = 1;
        int next_depth_tail = 1;
        int depth = 0;
        queue[tail++] = start;
        visited[start] = 1;

        while (head < tail) {
            const int v = queue[head++];
            order[order_len++] = v;

            int nnb = 0;
            for (int p = S->colptr[v]; p < S->colptr[v + 1]; ++p) {
                const int w = S->rowind[p];
                if (w != v && !visited[w]) {
                    neighbors[nnb++] = (Neighbor){.v = w, .degree = degree[w]};
                    visited[w] = 1;
                }
            }
            qsort(neighbors, (size_t)nnb, sizeof(Neighbor), neighbor_cmp);
            for (int i = 0; i < nnb; ++i) {
                queue[tail++] = neighbors[i].v;
            }
            next_depth_tail += nnb;

            if (head == depth_tail) {
                depth += 1;
                depth_tail = next_depth_tail;
            }
        }
        if (depth > max_component_depth) {
            max_component_depth = depth;
        }
    }

    int *old_at_new = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    for (int new_i = 0; new_i < n; ++new_i) {
        old_at_new[new_i] = order[n - 1 - new_i];
    }
    invert_permutation_old_to_new(old_at_new, old_to_new, n);

    *levels_hint = max_component_depth;
    free(old_at_new);
    free(neighbors);
    free(order);
    free(queue);
    free(visited);
    free(degree);
    csc_free(S);
    return 0;
}

static CscMatrix *apply_rcm(
    const CscMatrix *A,
    PreprocessTimings *timings,
    PreprocessTransform *transform)
{
    const int n = A->ncols;
    int *old_to_new = (int *)ilu_xmalloc((size_t)n * sizeof(int));

    const double t_order = wall_seconds();
    compute_rcm_old_to_new(A, old_to_new, &timings->rcm_levels_hint);
    timings->rcm_order += wall_seconds() - t_order;

    const double t_apply = wall_seconds();
    CscMatrix *B = csc_permute_symmetric(A, old_to_new);
    timings->rcm_apply += wall_seconds() - t_apply;
    timings->rcm = timings->rcm_order + timings->rcm_apply;
    transform_apply_symmetric_perm(transform, old_to_new);

    free(old_to_new);
    return B;
}

static CscMatrix *apply_symmetric_diag_scale(
    const CscMatrix *A,
    PreprocessTimings *timings,
    PreprocessTransform *transform)
{
    if (A->nrows != A->ncols) {
        fprintf(stderr, "diagonal scaling requires a square matrix\n");
        return NULL;
    }

    const double t0 = wall_seconds();
    const int n = A->ncols;
    double *d = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    for (int j = 0; j < n; ++j) {
        const double ajj = csc_diag_value(A, j);
        if (!(ajj > 0.0) || !isfinite(ajj)) {
            fprintf(stderr,
                    "diagonal scaling requires positive finite diagonal entries; "
                    "column %d has %.17g\n",
                    j, ajj);
            free(d);
            return NULL;
        }
        d[j] = 1.0 / sqrt(ajj);
    }

    CscMatrix *B = csc_alloc(A->nrows, A->ncols, A->nnz);
    memcpy(B->colptr, A->colptr, ((size_t)n + 1u) * sizeof(int));
    memcpy(B->rowind, A->rowind, (size_t)A->nnz * sizeof(int));
    for (int j = 0; j < n; ++j) {
        const double dj = d[j];
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            B->x[p] = A->x[p] * d[A->rowind[p]] * dj;
        }
    }
    transform_apply_current_diag_scale(transform, d);
    free(d);
    timings->diag_scale += wall_seconds() - t0;
    return B;
}

static CscMatrix *apply_left_diag_scale(
    const CscMatrix *A,
    PreprocessTimings *timings,
    PreprocessTransform *transform)
{
    if (A->nrows != A->ncols) {
        fprintf(stderr, "left diagonal scaling requires a square matrix\n");
        return NULL;
    }

    const double t0 = wall_seconds();
    const int n = A->ncols;
    double *d = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        const double aii = csc_diag_value(A, i);
        if (!(aii > 0.0) || !isfinite(aii)) {
            fprintf(stderr,
                    "left diagonal scaling requires positive finite diagonal entries; "
                    "row %d has %.17g\n",
                    i, aii);
            free(d);
            return NULL;
        }
        d[i] = 1.0 / aii;
    }

    CscMatrix *B = csc_alloc(A->nrows, A->ncols, A->nnz);
    memcpy(B->colptr, A->colptr, ((size_t)n + 1u) * sizeof(int));
    memcpy(B->rowind, A->rowind, (size_t)A->nnz * sizeof(int));
    for (int j = 0; j < n; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            B->x[p] = d[A->rowind[p]] * A->x[p];
        }
    }
    transform_apply_current_left_diag_scale(transform, d);
    free(d);
    timings->diag_scale += wall_seconds() - t0;
    return B;
}

#ifdef C_ILU_HAVE_MC64
static int diag_nonzeros_after_rowperm(const CscMatrix *A, const int *row_old_to_new)
{
    int count = 0;
    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            if (row_old_to_new[A->rowind[p]] == j && A->x[p] != 0.0) {
                count += 1;
                break;
            }
        }
    }
    return count;
}

static int validate_permutation(const int *p, int n)
{
    int *seen = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (p[i] < 0 || p[i] >= n || seen[p[i]]) {
            ok = 0;
            break;
        }
        seen[p[i]] = 1;
    }
    free(seen);
    return ok;
}

static CscMatrix *apply_mc64(
    const CscMatrix *A,
    PreprocessTimings *timings,
    PreprocessTransform *transform)
{
    const int n = A->ncols;
    if (A->nrows != A->ncols) {
        fprintf(stderr, "MC64 preprocessing requires a square matrix\n");
        return NULL;
    }

    const double t_prepare = wall_seconds();
    int n_mc64 = n;
    int_t *colptr = (int_t *)ilu_xmalloc(((size_t)n + 1) * sizeof(int_t));
    int_t *rowind = (int_t *)ilu_xmalloc((size_t)A->nnz * sizeof(int_t));
    int_t *cperm = (int_t *)ilu_xmalloc((size_t)n * sizeof(int_t));

    for (int i = 0; i <= n; ++i) {
        colptr[i] = (int_t)A->colptr[i] + 1;
    }
    for (int p = 0; p < A->nnz; ++p) {
        rowind[p] = (int_t)A->rowind[p] + 1;
    }

    int job = 5;
    int num = 0;
    int icntl[10];
    int info[10];
    int_t ne = (int_t)A->nnz;
    int_t liw = 5 * n;
    int_t ldw = 3 * n + ne;
    int_t *iw = (int_t *)ilu_xmalloc((size_t)liw * sizeof(int_t));
    double *dw = (double *)ilu_xmalloc((size_t)ldw * sizeof(double));

    mc64id_dist(icntl);
    icntl[0] = -1;
    icntl[1] = -1;
    timings->mc64_prepare += wall_seconds() - t_prepare;

    const double t_match = wall_seconds();
    mc64ad_dist(&job, &n_mc64, &ne, colptr, rowind, A->x, &num, cperm,
                &liw, iw, &ldw, dw, icntl, info);
    timings->mc64_match += wall_seconds() - t_match;

    if (info[0] < 0) {
        fprintf(stderr, "mc64ad_dist failed with info[0]=%d info[1]=%d\n", info[0], info[1]);
        free(dw);
        free(iw);
        free(cperm);
        free(rowind);
        free(colptr);
        return NULL;
    }
    if (num < n) {
        fprintf(stderr, "mc64ad_dist warning: structurally singular matching, matched %d of %d\n", num, n);
    }

    const double t_apply = wall_seconds();
    int *row_old_to_new = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    for (int old_row = 0; old_row < n; ++old_row) {
        const int matched_old_col = (int)cperm[old_row] - 1;
        if (matched_old_col < 0 || matched_old_col >= n) {
            fprintf(stderr, "mc64ad_dist returned an invalid column permutation\n");
            free(row_old_to_new);
            free(dw);
            free(iw);
            free(cperm);
            free(rowind);
            free(colptr);
            return NULL;
        }
        row_old_to_new[old_row] = matched_old_col;
    }

    if (!validate_permutation(row_old_to_new, n)) {
        fprintf(stderr, "mc64ad_dist returned a non-permutation matching\n");
        free(row_old_to_new);
        free(dw);
        free(iw);
        free(cperm);
        free(rowind);
        free(colptr);
        return NULL;
    }

    CscMatrix *B = csc_permute_rows_scale(A, row_old_to_new, dw, dw + n);
    transform_apply_rowperm_log_scale(transform, row_old_to_new, dw, dw + n);
    timings->diag_nonzeros_after_mc64 = diag_nonzeros_after_rowperm(A, row_old_to_new);
    timings->mc64_apply += wall_seconds() - t_apply;
    timings->mc64 = timings->mc64_prepare + timings->mc64_match + timings->mc64_apply;

    free(row_old_to_new);
    free(dw);
    free(iw);
    free(cperm);
    free(rowind);
    free(colptr);
    return B;
}
#else
static CscMatrix *apply_mc64(
    const CscMatrix *A,
    PreprocessTimings *timings,
    PreprocessTransform *transform)
{
    (void)timings;
    (void)transform;
    preprocess_warn_unavailable();
    return csc_clone(A);
}
#endif

CscMatrix *preprocess_matrix(const CscMatrix *A, PreprocessMode mode, PreprocessTimings *timings)
{
    return preprocess_matrix_with_transform(A, mode, timings, NULL);
}

CscMatrix *preprocess_matrix_with_transform(
    const CscMatrix *A,
    PreprocessMode mode,
    PreprocessTimings *timings,
    PreprocessTransform *transform)
{
    memset(timings, 0, sizeof(*timings));
    timings->requested_mode = mode;
    timings->effective_mode = preprocess_effective_mode(mode);
    const double t0 = wall_seconds();

    CscMatrix *current = NULL;
    if (transform) {
        if (A->nrows != A->ncols) {
            fprintf(stderr, "preprocess transform requires a square matrix\n");
            return NULL;
        }
        preprocess_transform_init(transform, A->ncols);
    }
    if (mode == PREPROCESS_NONE) {
        timings->total = 0.0;
        return NULL;
    }

    if (mode == PREPROCESS_MC64 || mode == PREPROCESS_MC64_RCM) {
        current = apply_mc64(A, timings, transform);
        if (!current) {
            preprocess_transform_free(transform);
            return NULL;
        }
    }

    if (mode == PREPROCESS_DIAG || mode == PREPROCESS_DIAG_RCM) {
        current = apply_symmetric_diag_scale(A, timings, transform);
        if (!current) {
            preprocess_transform_free(transform);
            return NULL;
        }
    }

    if (mode == PREPROCESS_LEFT_DIAG) {
        current = apply_left_diag_scale(A, timings, transform);
        if (!current) {
            preprocess_transform_free(transform);
            return NULL;
        }
    }

    if (mode == PREPROCESS_LEFT_DIAG_RCM) {
        current = apply_rcm(A, timings, transform);
        if (!current) {
            preprocess_transform_free(transform);
            return NULL;
        }
        CscMatrix *scaled = apply_left_diag_scale(current, timings, transform);
        csc_free(current);
        current = scaled;
        if (!current) {
            preprocess_transform_free(transform);
            return NULL;
        }
    }

    if (mode == PREPROCESS_RCM || mode == PREPROCESS_MC64_RCM || mode == PREPROCESS_DIAG_RCM) {
        const CscMatrix *input = current ? current : A;
        /* For mc64-rcm, RCM is symmetric on the MC64-matched matrix, so
           each matched row/column pivot pair moves together. */
        CscMatrix *rcm = apply_rcm(input, timings, transform);
        csc_free(current);
        current = rcm;
        if (!current) {
            preprocess_transform_free(transform);
            return NULL;
        }
    }

    timings->total = wall_seconds() - t0;
    return current;
}
