#include "ilu.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int *idx;
    int *lvl;
    int len;
} PatternRow;

typedef struct {
    int n;
    int nnz;
    int *row_ptr;
    int *row_pattern;
} RowPattern;

static void min_heap_push(int *heap, int *size, int value)
{
    int pos = (*size)++;
    while (pos > 0) {
        const int parent = (pos - 1) / 2;
        if (heap[parent] <= value) {
            break;
        }
        heap[pos] = heap[parent];
        pos = parent;
    }
    heap[pos] = value;
}

static int min_heap_pop(int *heap, int *size)
{
    const int result = heap[0];
    const int value = heap[--(*size)];
    int pos = 0;

    while (1) {
        const int left = 2 * pos + 1;
        if (left >= *size) {
            break;
        }
        const int right = left + 1;
        const int child = right < *size && heap[right] < heap[left] ? right : left;
        if (heap[child] >= value) {
            break;
        }
        heap[pos] = heap[child];
        pos = child;
    }
    if (*size > 0) {
        heap[pos] = value;
    }
    return result;
}

static void pivot_work_push(
    int *work,
    int *size,
    int value,
    int require_order)
{
    if (require_order) {
        min_heap_push(work, size, value);
    } else {
        work[(*size)++] = value;
    }
}

static int pivot_work_pop(
    int *work,
    int *head,
    int *size,
    int require_order)
{
    return require_order ? min_heap_pop(work, size) : work[(*head)++];
}

static int pivot_work_pending(int head, int size, int require_order)
{
    return require_order ? size > 0 : head < size;
}

static void pattern_rows_free(PatternRow *rows, int n)
{
    if (!rows) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        free(rows[i].idx);
        free(rows[i].lvl);
    }
    free(rows);
}

static int build_row_pattern(const CscMatrix *A, RowPattern *rp)
{
    const int n = A->nrows;
    const int nnzA = A->nnz;
    int *row_counts = (int *)ilu_xcalloc((size_t)n, sizeof(int));

    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            row_counts[A->rowind[p]] += 1;
        }
    }

    rp->n = n;
    rp->nnz = nnzA;
    rp->row_ptr = (int *)ilu_xmalloc(((size_t)n + 1) * sizeof(int));
    rp->row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        rp->row_ptr[i + 1] = rp->row_ptr[i] + row_counts[i];
    }

    rp->row_pattern = (int *)ilu_xmalloc((size_t)nnzA * sizeof(int));
    memset(row_counts, 0, (size_t)n * sizeof(int));
    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            const int r = A->rowind[p];
            const int idx = rp->row_ptr[r] + row_counts[r]++;
            rp->row_pattern[idx] = j;
        }
    }

    free(row_counts);
    return 0;
}

static void row_pattern_free(RowPattern *rp)
{
    if (!rp) {
        return;
    }
    free(rp->row_ptr);
    free(rp->row_pattern);
    memset(rp, 0, sizeof(*rp));
}

static int build_symmetric_dependency_levels(const CscMatrix *A, int **level_out, int *nlevels_out)
{
    const int n = A->nrows;
    int *dep_counts = (int *)ilu_xcalloc((size_t)n, sizeof(int));

    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            const int r = A->rowind[p];
            if (r == j) {
                continue;
            }
            const int high = r > j ? r : j;
            dep_counts[high] += 1;
        }
    }

    int *dep_ptr = (int *)ilu_xmalloc(((size_t)n + 1) * sizeof(int));
    dep_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        dep_ptr[i + 1] = dep_ptr[i] + dep_counts[i];
    }

    int *deps = (int *)ilu_xmalloc((size_t)dep_ptr[n] * sizeof(int));
    memset(dep_counts, 0, (size_t)n * sizeof(int));
    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            const int r = A->rowind[p];
            if (r == j) {
                continue;
            }
            const int high = r > j ? r : j;
            const int low = r > j ? j : r;
            deps[dep_ptr[high] + dep_counts[high]++] = low;
        }
    }

    int *levels = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    int max_level = 0;
    for (int i = 0; i < n; ++i) {
        int li = 0;
        for (int p = dep_ptr[i]; p < dep_ptr[i + 1]; ++p) {
            const int d = deps[p];
            const int candidate = levels[d] + 1;
            if (candidate > li) {
                li = candidate;
            }
        }
        levels[i] = li;
        if (li > max_level) {
            max_level = li;
        }
    }

    free(deps);
    free(dep_ptr);
    free(dep_counts);

    *level_out = levels;
    *nlevels_out = max_level + 1;
    return 0;
}

static int symbolic_ilu0(const CscMatrix *A, CscMatrix **Lout, CscMatrix **Uout)
{
    int lnnz = 0;
    int unnz = 0;
    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            const int r = A->rowind[p];
            lnnz += r >= j;
            unnz += r <= j;
        }
    }

    CscMatrix *L = csc_alloc(A->nrows, A->ncols, lnnz);
    CscMatrix *U = csc_alloc(A->nrows, A->ncols, unnz);
    int lp = 0;
    int up = 0;
    L->colptr[0] = 0;
    U->colptr[0] = 0;
    for (int j = 0; j < A->ncols; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            const int r = A->rowind[p];
            if (r >= j) {
                L->rowind[lp] = r;
                L->x[lp] = 1.0;
                lp += 1;
            }
            if (r <= j) {
                U->rowind[up] = r;
                U->x[up] = 1.0;
                up += 1;
            }
        }
        L->colptr[j + 1] = lp;
        U->colptr[j + 1] = up;
    }

    *Lout = L;
    *Uout = U;
    return 0;
}

static int symbolic_ilu_k(const CscMatrix *A, int k, CscMatrix **Lout, CscMatrix **Uout)
{
    if (A->nrows != A->ncols) {
        fprintf(stderr, "symbolic ILU requires a square matrix\n");
        return -1;
    }
    if (k == 0) {
        return symbolic_ilu0(A, Lout, Uout);
    }

    const int n = A->nrows;
    const int nnzA = A->nnz;
    const int require_order = k >= 3;

    int *row_counts = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    for (int j = 0; j < n; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            row_counts[A->rowind[p]] += 1;
        }
    }

    int *row_ptr = (int *)ilu_xmalloc(((size_t)n + 1) * sizeof(int));
    row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
    }

    int *row_pattern = (int *)ilu_xmalloc((size_t)nnzA * sizeof(int));
    int *temp = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    for (int j = 0; j < n; ++j) {
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            const int r = A->rowind[p];
            const int idx = row_ptr[r] + temp[r]++;
            row_pattern[idx] = j;
        }
    }

    int *countsL = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    int *countsU = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    int *level = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    int *heap = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    int *touched = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    PatternRow *Urows = (PatternRow *)ilu_xcalloc((size_t)n, sizeof(PatternRow));

    for (int i = 0; i < n; ++i) {
        level[i] = k + 1;
    }

    for (int i = 0; i < n; ++i) {
        int touched_t = 0;
        int heap_head = 0;
        int heap_size = 0;

        for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            const int c = row_pattern[idx];
            if (level[c] == k + 1) {
                touched[touched_t++] = c;
                level[c] = 0;
                if (c < i) {
                    pivot_work_push(heap, &heap_size, c, require_order);
                }
            }
        }

        while (pivot_work_pending(heap_head, heap_size, require_order)) {
            const int j = pivot_work_pop(
                heap, &heap_head, &heap_size, require_order);
            const int lvlj = level[j];
            for (int q = 0; q < Urows[j].len; ++q) {
                const int w = Urows[j].idx[q];
                const int wlvl = Urows[j].lvl[q];
                const int new_level = lvlj + wlvl + 1;
                if (new_level <= k && new_level < level[w]) {
                    if (level[w] == k + 1) {
                        touched[touched_t++] = w;
                        if (w < i) {
                            pivot_work_push(heap, &heap_size, w, require_order);
                        }
                    }
                    level[w] = new_level;
                }
            }
        }

        int ucount = 0;
        for (int t = 0; t < touched_t; ++t) {
            const int c = touched[t];
            const int lvl = level[c];
            if (lvl <= k) {
                if (c <= i) {
                    countsL[c] += 1;
                }
                if (c >= i) {
                    countsU[c] += 1;
                    ucount += 1;
                }
            }
        }

        Urows[i].idx = (int *)ilu_xmalloc((size_t)ucount * sizeof(int));
        Urows[i].lvl = (int *)ilu_xmalloc((size_t)ucount * sizeof(int));
        Urows[i].len = ucount;
        int pos = 0;
        for (int t = 0; t < touched_t; ++t) {
            const int c = touched[t];
            const int lvl = level[c];
            if (lvl <= k && c >= i) {
                Urows[i].idx[pos] = c;
                Urows[i].lvl[pos] = lvl;
                pos += 1;
            }
            level[c] = k + 1;
        }
    }

    CscMatrix *L = csc_alloc(n, n, 0);
    CscMatrix *U = csc_alloc(n, n, 0);
    free(L->rowind);
    free(L->x);
    free(U->rowind);
    free(U->x);

    L->colptr[0] = 0;
    U->colptr[0] = 0;
    for (int j = 0; j < n; ++j) {
        L->colptr[j + 1] = L->colptr[j] + countsL[j];
        U->colptr[j + 1] = U->colptr[j] + countsU[j];
    }
    L->nnz = L->colptr[n];
    U->nnz = U->colptr[n];
    L->rowind = (int *)ilu_xmalloc((size_t)L->nnz * sizeof(int));
    U->rowind = (int *)ilu_xmalloc((size_t)U->nnz * sizeof(int));
    L->x = (double *)ilu_xmalloc((size_t)L->nnz * sizeof(double));
    U->x = (double *)ilu_xmalloc((size_t)U->nnz * sizeof(double));
    csc_set_all(L, 1.0);
    csc_set_all(U, 1.0);

    int *nextL = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    int *nextU = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    memcpy(nextL, L->colptr, (size_t)n * sizeof(int));
    memcpy(nextU, U->colptr, (size_t)n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        level[i] = k + 1;
    }

    for (int i = 0; i < n; ++i) {
        int touched_t = 0;
        int heap_head = 0;
        int heap_size = 0;

        for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            const int c = row_pattern[idx];
            if (level[c] == k + 1) {
                touched[touched_t++] = c;
                level[c] = 0;
                if (c < i) {
                    pivot_work_push(heap, &heap_size, c, require_order);
                }
            }
        }

        while (pivot_work_pending(heap_head, heap_size, require_order)) {
            const int j = pivot_work_pop(
                heap, &heap_head, &heap_size, require_order);
            const int lvlj = level[j];
            for (int q = 0; q < Urows[j].len; ++q) {
                const int w = Urows[j].idx[q];
                const int wlvl = Urows[j].lvl[q];
                const int new_level = lvlj + wlvl + 1;
                if (new_level <= k && new_level < level[w]) {
                    if (level[w] == k + 1) {
                        touched[touched_t++] = w;
                        if (w < i) {
                            pivot_work_push(heap, &heap_size, w, require_order);
                        }
                    }
                    level[w] = new_level;
                }
            }
        }

        for (int t = 0; t < touched_t; ++t) {
            const int c = touched[t];
            const int lvl = level[c];
            if (lvl <= k) {
                if (c <= i) {
                    const int p = nextL[c]++;
                    L->rowind[p] = i;
                }
                if (c >= i) {
                    const int p = nextU[c]++;
                    U->rowind[p] = i;
                }
            }
            level[c] = k + 1;
        }
    }

    for (int i = 0; i < n; ++i) {
        free(Urows[i].idx);
        free(Urows[i].lvl);
    }
    free(Urows);
    free(nextU);
    free(nextL);
    free(touched);
    free(heap);
    free(level);
    free(countsU);
    free(countsL);
    free(temp);
    free(row_pattern);
    free(row_ptr);
    free(row_counts);

    *Lout = L;
    *Uout = U;
    return 0;
}

static int symbolic_ilu_k_levelset(
    const CscMatrix *A,
    int k,
    CscMatrix **Lout,
    CscMatrix **Uout,
    int *nlevels_out,
    int *fallback_out)
{
    *nlevels_out = 0;
    *fallback_out = 0;

    if (A->nrows != A->ncols) {
        fprintf(stderr, "symbolic ILU requires a square matrix\n");
        return -1;
    }
    if (k == 0) {
        *nlevels_out = 1;
        return symbolic_ilu0(A, Lout, Uout);
    }

    const int n = A->nrows;
    const int require_order = k >= 3;
    RowPattern rp = {0};
    int *row_levels = NULL;
    int nlevels = 0;
    build_row_pattern(A, &rp);
    build_symmetric_dependency_levels(A, &row_levels, &nlevels);
    *nlevels_out = nlevels;

    int *level_counts = (int *)ilu_xcalloc((size_t)nlevels, sizeof(int));
    for (int i = 0; i < n; ++i) {
        level_counts[row_levels[i]] += 1;
    }

    int *level_ptr = (int *)ilu_xmalloc(((size_t)nlevels + 1) * sizeof(int));
    level_ptr[0] = 0;
    for (int l = 0; l < nlevels; ++l) {
        level_ptr[l + 1] = level_ptr[l] + level_counts[l];
    }

    int *next_level = (int *)ilu_xmalloc((size_t)nlevels * sizeof(int));
    memcpy(next_level, level_ptr, (size_t)nlevels * sizeof(int));
    int *level_rows = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        const int l = row_levels[i];
        level_rows[next_level[l]++] = i;
    }

    PatternRow *Lrows = (PatternRow *)ilu_xcalloc((size_t)n, sizeof(PatternRow));
    PatternRow *Urows = (PatternRow *)ilu_xcalloc((size_t)n, sizeof(PatternRow));
    int *countsL = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    int *countsU = (int *)ilu_xcalloc((size_t)n, sizeof(int));
    int unsafe = 0;

    for (int l = 0; l < nlevels && !unsafe; ++l) {
        const int start = level_ptr[l];
        const int end = level_ptr[l + 1];

#pragma omp parallel if(end - start >= 8)
        {
            int *level = (int *)ilu_xmalloc((size_t)n * sizeof(int));
            int *heap = (int *)ilu_xmalloc((size_t)n * sizeof(int));
            int *touched = (int *)ilu_xmalloc((size_t)n * sizeof(int));
            for (int i = 0; i < n; ++i) {
                level[i] = k + 1;
            }

#pragma omp for schedule(dynamic)
            for (int pos = start; pos < end; ++pos) {
                int local_unsafe = 0;
                const int i = level_rows[pos];
                int touched_t = 0;
                int heap_head = 0;
                int heap_size = 0;

                for (int idx = rp.row_ptr[i]; idx < rp.row_ptr[i + 1]; ++idx) {
                    const int c = rp.row_pattern[idx];
                    if (level[c] == k + 1) {
                        touched[touched_t++] = c;
                        level[c] = 0;
                        if (c < i) {
                            pivot_work_push(heap, &heap_size, c, require_order);
                        }
                    }
                }

                while (pivot_work_pending(heap_head, heap_size, require_order)) {
                    const int j = pivot_work_pop(
                        heap, &heap_head, &heap_size, require_order);
                    const int lvlj = level[j];
                    for (int q = 0; q < Urows[j].len; ++q) {
                        const int w = Urows[j].idx[q];
                        const int wlvl = Urows[j].lvl[q];
                        const int new_level = lvlj + wlvl + 1;
                        if (new_level <= k && new_level < level[w]) {
                            if (level[w] == k + 1) {
                                touched[touched_t++] = w;
                                if (w < i) {
                                    if (row_levels[w] >= row_levels[i]) {
                                        local_unsafe = 1;
                                    } else {
                                        pivot_work_push(
                                            heap, &heap_size, w, require_order);
                                    }
                                }
                            }
                            level[w] = new_level;
                        }
                    }
                }

                if (local_unsafe) {
#pragma omp atomic write
                    unsafe = 1;
                }

                int lcount = 0;
                int ucount = 0;
                for (int t = 0; t < touched_t; ++t) {
                    const int c = touched[t];
                    const int lvl = level[c];
                    if (lvl <= k) {
                        if (c <= i) {
                            lcount += 1;
                        }
                        if (c >= i) {
                            ucount += 1;
                        }
                    }
                }

                Lrows[i].idx = (int *)ilu_xmalloc((size_t)lcount * sizeof(int));
                Lrows[i].len = lcount;
                Urows[i].idx = (int *)ilu_xmalloc((size_t)ucount * sizeof(int));
                Urows[i].lvl = (int *)ilu_xmalloc((size_t)ucount * sizeof(int));
                Urows[i].len = ucount;

                int lpos = 0;
                int upos = 0;
                for (int t = 0; t < touched_t; ++t) {
                    const int c = touched[t];
                    const int lvl = level[c];
                    if (lvl <= k) {
                        if (c <= i) {
                            Lrows[i].idx[lpos++] = c;
#pragma omp atomic update
                            countsL[c] += 1;
                        }
                        if (c >= i) {
                            Urows[i].idx[upos] = c;
                            Urows[i].lvl[upos] = lvl;
                            upos += 1;
#pragma omp atomic update
                            countsU[c] += 1;
                        }
                    }
                    level[c] = k + 1;
                }
            }

            free(touched);
            free(heap);
            free(level);
        }
    }

    free(next_level);
    free(level_rows);
    free(level_ptr);
    free(level_counts);
    row_pattern_free(&rp);

    if (unsafe) {
        *fallback_out = 1;
        pattern_rows_free(Lrows, n);
        pattern_rows_free(Urows, n);
        free(countsU);
        free(countsL);
        free(row_levels);
        return symbolic_ilu_k(A, k, Lout, Uout);
    }

    CscMatrix *L = csc_alloc(n, n, 0);
    CscMatrix *U = csc_alloc(n, n, 0);
    free(L->rowind);
    free(L->x);
    free(U->rowind);
    free(U->x);

    L->colptr[0] = 0;
    U->colptr[0] = 0;
    for (int j = 0; j < n; ++j) {
        L->colptr[j + 1] = L->colptr[j] + countsL[j];
        U->colptr[j + 1] = U->colptr[j] + countsU[j];
    }
    L->nnz = L->colptr[n];
    U->nnz = U->colptr[n];
    L->rowind = (int *)ilu_xmalloc((size_t)L->nnz * sizeof(int));
    U->rowind = (int *)ilu_xmalloc((size_t)U->nnz * sizeof(int));
    L->x = (double *)ilu_xmalloc((size_t)L->nnz * sizeof(double));
    U->x = (double *)ilu_xmalloc((size_t)U->nnz * sizeof(double));
    csc_set_all(L, 1.0);
    csc_set_all(U, 1.0);

    int *nextL = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    int *nextU = (int *)ilu_xmalloc((size_t)n * sizeof(int));
    memcpy(nextL, L->colptr, (size_t)n * sizeof(int));
    memcpy(nextU, U->colptr, (size_t)n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int p = 0; p < Lrows[i].len; ++p) {
            const int c = Lrows[i].idx[p];
            L->rowind[nextL[c]++] = i;
        }
        for (int p = 0; p < Urows[i].len; ++p) {
            const int c = Urows[i].idx[p];
            U->rowind[nextU[c]++] = i;
        }
    }

    free(nextU);
    free(nextL);
    pattern_rows_free(Lrows, n);
    pattern_rows_free(Urows, n);
    free(countsU);
    free(countsL);
    free(row_levels);

    *Lout = L;
    *Uout = U;
    return 0;
}

static void fill_symbolic_values(const CscMatrix *A, CscMatrix *L, CscMatrix *U)
{
    const int n = A->ncols;
    for (int j = 0; j < n; ++j) {
        int pa = A->colptr[j];
        const int pa_end = A->colptr[j + 1];
        int pl = L->colptr[j];
        const int pl_end = L->colptr[j + 1];
        int pu = U->colptr[j];
        const int pu_end = U->colptr[j + 1];

        while (pa < pa_end) {
            const int ra = A->rowind[pa];
            if (ra <= j) {
                while (pu < pu_end && U->rowind[pu] < ra) {
                    pu += 1;
                }
                if (pu < pu_end && U->rowind[pu] == ra) {
                    U->x[pu] = A->x[pa];
                }
            }
            if (ra >= j) {
                while (pl < pl_end && L->rowind[pl] < ra) {
                    pl += 1;
                }
                if (pl < pl_end && L->rowind[pl] == ra) {
                    L->x[pl] = A->x[pa];
                }
            }
            pa += 1;
        }
    }
}

static void copy_to_symbol_values(const CscMatrix *A, CscMatrix *S)
{
    const int n = A->ncols;
    for (int j = 0; j < n; ++j) {
        int pA = A->colptr[j];
        const int pAend = A->colptr[j + 1];
        int pS = S->colptr[j];
        const int pSend = S->colptr[j + 1];

        while (pA < pAend && pS < pSend) {
            const int rA = A->rowind[pA];
            const int rS = S->rowind[pS];
            if (rA == rS) {
                S->x[pS] = A->x[pA];
                pA += 1;
                pS += 1;
            } else if (rA < rS) {
                pA += 1;
            } else {
                pS += 1;
            }
        }
    }
}

int ilu_build_symbolic_mode(
    const CscMatrix *A,
    int k,
    IluSymbolicMode mode,
    IluSymbolic *sym,
    IluSymbolicTimings *timings)
{
    memset(sym, 0, sizeof(*sym));
    memset(timings, 0, sizeof(*timings));

    const double t0 = wall_seconds();
    CscMatrix *Lraw = NULL;
    CscMatrix *Uraw = NULL;
    const double ts0 = wall_seconds();
    int symbolic_status = 0;
    if (mode == ILU_SYMBOLIC_LEVELSET) {
        symbolic_status = symbolic_ilu_k_levelset(
            A, k, &Lraw, &Uraw, &timings->symbolic_levels, &timings->symbolic_fallback);
    } else {
        timings->symbolic_levels = 0;
        timings->symbolic_fallback = 0;
        symbolic_status = symbolic_ilu_k(A, k, &Lraw, &Uraw);
    }
    if (symbolic_status != 0) {
        return -1;
    }
    timings->symbolic_pattern = wall_seconds() - ts0;

    const double tf0 = wall_seconds();
    sym->S = csc_union_pattern(Lraw, Uraw);
    csc_zero(Lraw);
    csc_zero(Uraw);
    fill_symbolic_values(A, Lraw, Uraw);
    copy_to_symbol_values(A, sym->S);
    timings->numeric_fill = wall_seconds() - tf0;

    const double tt0 = wall_seconds();
    sym->Lsym = csc_transpose_keepzeros(Lraw);
    sym->Usym = Uraw;
    sym->Lsymm1 = csc_triu_strict(sym->Lsym);
    csc_free(Lraw);
    timings->transform = wall_seconds() - tt0;
    timings->total = wall_seconds() - t0;

    if (!sym->S || !csc_validate_sorted(sym->Lsym) || !csc_validate_sorted(sym->Usym) ||
        !csc_validate_sorted(sym->S) || !csc_validate_sorted(sym->Lsymm1)) {
        fprintf(stderr, "symbolic setup produced invalid CSC data\n");
        ilu_symbolic_free(sym);
        return -1;
    }

    return 0;
}

int ilu_build_symbolic(const CscMatrix *A, int k, IluSymbolic *sym, IluSymbolicTimings *timings)
{
    return ilu_build_symbolic_mode(A, k, ILU_SYMBOLIC_LEVELSET, sym, timings);
}

void ilu_symbolic_free(IluSymbolic *sym)
{
    if (!sym) {
        return;
    }
    csc_free(sym->Lsym);
    csc_free(sym->Usym);
    csc_free(sym->S);
    csc_free(sym->Lsymm1);
    memset(sym, 0, sizeof(*sym));
}
