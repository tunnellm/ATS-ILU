#include "ilu.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARILU_OMP_CHUNK 4096
#define PARIC_OMP_CHUNK 4096
#define ATS_IC_OMP_CHUNK 64

#ifndef ATS_ASYNC_OMP_SCHEDULE
#define ATS_ASYNC_OMP_SCHEDULE static
#endif

#ifdef ATS_ASYNC_OMP_CHUNK
#define ATS_ASYNC_OMP_FOR_SCHEDULE schedule(ATS_ASYNC_OMP_SCHEDULE, ATS_ASYNC_OMP_CHUNK)
#else
#define ATS_ASYNC_OMP_FOR_SCHEDULE schedule(ATS_ASYNC_OMP_SCHEDULE)
#endif

static inline int binary_search_index(
    const int * restrict arr,
    int value,
    int start,
    int end)
{
    int lo = start;
    int hi = end;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (arr[mid] == value) {
            return mid;
        }
        if (arr[mid] < value) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

static inline double intersection_dot_onfly(
    int xpos,
    int xend,
    const int * restrict Xi,
    const double * restrict Xv,
    int ypos,
    int yend,
    const int * restrict Yi,
    const double * restrict Yv)
{
    double dot = 0.0;
    while (xpos < xend && ypos < yend) {
        const int xi = Xi[xpos];
        const int yi = Yi[ypos];
        if (xi == yi) {
            dot += Xv[xpos] * Yv[ypos];
            xpos += 1;
            ypos += 1;
        } else if (xi < yi) {
            xpos += 1;
        } else {
            ypos += 1;
        }
    }
    return dot;
}

static inline double intersection_dot_onfly_norestrict(
    int xpos,
    int xend,
    const int * Xi,
    const double * Xv,
    int ypos,
    int yend,
    const int * Yi,
    const double * Yv)
{
    double dot = 0.0;
    while (xpos < xend && ypos < yend) {
        const int xi = Xi[xpos];
        const int yi = Yi[ypos];
        if (xi == yi) {
            dot += Xv[xpos] * Yv[ypos];
            xpos += 1;
            ypos += 1;
        } else if (xi < yi) {
            xpos += 1;
        } else {
            ypos += 1;
        }
    }
    return dot;
}

static inline double async_load_factor_value(const double *value)
{
    double loaded;
    __atomic_load(value, &loaded, __ATOMIC_RELAXED);
    return loaded;
}

static inline void async_publish_factor_row(
    double *dst,
    const double *src,
    int count)
{
    for (int i = 0; i < count; ++i) {
        __atomic_store(dst + i, src + i, __ATOMIC_RELAXED);
    }
}

static inline double intersection_dot_onfly_async(
    int xpos,
    int xend,
    const int *Xi,
    const double *Xv,
    int ypos,
    int yend,
    const int *Yi,
    const double *Yv)
{
    double dot = 0.0;
    while (xpos < xend && ypos < yend) {
        const int xi = Xi[xpos];
        const int yi = Yi[ypos];
        if (xi == yi) {
            dot += Xv[xpos] * async_load_factor_value(Yv + ypos);
            xpos += 1;
            ypos += 1;
        } else if (xi < yi) {
            xpos += 1;
        } else {
            ypos += 1;
        }
    }
    return dot;
}

static inline double intersection_dot_onfly_all_async(
    int xpos,
    int xend,
    const int *Xi,
    const double *Xv,
    int ypos,
    int yend,
    const int *Yi,
    const double *Yv)
{
    double dot = 0.0;
    while (xpos < xend && ypos < yend) {
        const int xi = Xi[xpos];
        const int yi = Yi[ypos];
        if (xi == yi) {
            dot += async_load_factor_value(Xv + xpos) *
                async_load_factor_value(Yv + ypos);
            xpos += 1;
            ypos += 1;
        } else if (xi < yi) {
            xpos += 1;
        } else {
            ypos += 1;
        }
    }
    return dot;
}

static inline double intersection_dot_scaled_unitdiag_onfly(
    int xpos,
    int xend,
    const int * restrict Xi,
    const double * restrict Xv,
    int ypos,
    int yend,
    const int * restrict Yi,
    const double * restrict Yv,
    const double * restrict row_scale)
{
    double dot = 0.0;
    while (xpos < xend && ypos < yend) {
        const int xi = Xi[xpos];
        const int yi = Yi[ypos];
        if (xi == yi) {
            dot += Xv[xpos] * (row_scale[yi] * Yv[ypos]);
            xpos += 1;
            ypos += 1;
        } else if (xi < yi) {
            xpos += 1;
        } else {
            ypos += 1;
        }
    }
    return dot;
}

static void triangular_solve_row_onfly(
    int i,
    const int * restrict Xp,
    const int * restrict Xi,
    double * restrict Xv,
    const int * restrict Yp,
    const int * restrict Yi,
    const double * restrict Yv)
{
    for (int idx = Xp[i]; idx < Xp[i + 1]; ++idx) {
        const int j = Xi[idx];
        const int diag = Yp[j + 1] - 1;
        const double dot = intersection_dot_onfly(
            Xp[i], idx, Xi, Xv,
            Yp[j], diag, Yi, Yv);
        const double diag_inv = 1.0 / Yv[diag];
        Xv[idx] = (Xv[idx] - dot) * diag_inv;
    }
}

static void triangular_solve_row_rhs_onfly(
    int i,
    const int * restrict Xp,
    const int * restrict Xi,
    const double * restrict Xrhs,
    double * restrict Xv,
    const int * restrict Yp,
    const int * restrict Yi,
    const double * restrict Yv)
{
    for (int idx = Xp[i]; idx < Xp[i + 1]; ++idx) {
        const int j = Xi[idx];
        const int diag = Yp[j + 1] - 1;
        const double dot = intersection_dot_onfly(
            Xp[i], idx, Xi, Xv,
            Yp[j], diag, Yi, Yv);
        const double diag_inv = 1.0 / Yv[diag];
        Xv[idx] = (Xrhs[idx] - dot) * diag_inv;
    }
}

static void triangular_solve_row_rhs_onfly_norestrict(
    int i,
    const int *Xp,
    const int *Xi,
    const double *Xrhs,
    double *Xv,
    const int *Yp,
    const int *Yi,
    const double *Yv)
{
    for (int idx = Xp[i]; idx < Xp[i + 1]; ++idx) {
        const int j = Xi[idx];
        const int diag = Yp[j + 1] - 1;
        const double dot = intersection_dot_onfly_async(
            Xp[i], idx, Xi, Xv,
            Yp[j], diag, Yi, Yv);
        const double diag_inv = 1.0 / async_load_factor_value(Yv + diag);
        Xv[idx] = (Xrhs[idx] - dot) * diag_inv;
    }
}

static void triangular_solve_row_rhs_unitdiag_onfly_norestrict(
    int i,
    const int *Xp,
    const int *Xi,
    const double *Xrhs,
    double *Xv,
    const int *Yp,
    const int *Yi,
    const double *Yv)
{
    const int diagonal = Xp[i + 1] - 1;
    for (int idx = Xp[i]; idx < diagonal; ++idx) {
        const int j = Xi[idx];
        const int ydiag = Yp[j + 1] - 1;
        const double dot = intersection_dot_onfly_async(
            Xp[i], idx, Xi, Xv,
            Yp[j], ydiag, Yi, Yv);
        const double diag_inv = 1.0 / async_load_factor_value(Yv + ydiag);
        Xv[idx] = (Xrhs[idx] - dot) * diag_inv;
    }
    Xv[diagonal] = 1.0;
}

static void triangular_solve_row_rhs_scaled_unitdiag_onfly(
    int i,
    const int * restrict Xp,
    const int * restrict Xi,
    const double * restrict Xrhs,
    double * restrict Xv,
    const int * restrict Yp,
    const int * restrict Yi,
    const double * restrict Yv,
    const double * restrict row_scale)
{
    for (int idx = Xp[i]; idx < Xp[i + 1]; ++idx) {
        const int j = Xi[idx];
        const int diag = Yp[j + 1] - 1;
        const double dot = intersection_dot_scaled_unitdiag_onfly(
            Xp[i], idx, Xi, Xv,
            Yp[j], diag, Yi, Yv, row_scale);
        Xv[idx] = Xrhs[idx] - dot;
    }
}

double ilu_workspace_checksum(const CscMatrix *L, const CscMatrix *U)
{
    return csc_abs_checksum(L) + csc_abs_checksum(U);
}

int seq_ilu_setup(const IluSymbolic *sym, SeqIluWorkspace *w)
{
    memset(w, 0, sizeof(*w));
    w->L = csc_clone(sym->Lsym);
    w->U = csc_clone(sym->Usym);
    return 0;
}

void seq_ilu_free(SeqIluWorkspace *w)
{
    if (!w) {
        return;
    }
    csc_free(w->L);
    csc_free(w->U);
    memset(w, 0, sizeof(*w));
}

void seq_ilu_factor(SeqIluWorkspace *w)
{
    const int n = w->L->ncols;
    for (int i = 0; i < n; ++i) {
        triangular_solve_row_onfly(
            i, w->L->colptr, w->L->rowind, w->L->x,
            w->U->colptr, w->U->rowind, w->U->x);
        triangular_solve_row_onfly(
            i, w->U->colptr, w->U->rowind, w->U->x,
            w->L->colptr, w->L->rowind, w->L->x);
    }
}

static int ats_ilu_setup_common(const IluSymbolic *sym, AtsIluWorkspace *w, int allocate_tmp)
{
    memset(w, 0, sizeof(*w));
    w->Lsrc_mat = sym->Lsym;
    w->Usrc_mat = sym->Usym;
    w->Lsrc = sym->Lsym->x;
    w->Usrc = sym->Usym->x;
    w->L = csc_clone(sym->Lsym);
    w->U = csc_clone(sym->Usym);
    w->D = (double *)ilu_xmalloc((size_t)w->L->ncols * sizeof(double));
    if (allocate_tmp) {
        w->Ltmp = (double *)ilu_xmalloc((size_t)w->L->nnz * sizeof(double));
        w->Utmp = (double *)ilu_xmalloc((size_t)w->U->nnz * sizeof(double));
    }
    return 0;
}

int ats_ilu_setup(const IluSymbolic *sym, AtsIluWorkspace *w)
{
    return ats_ilu_setup_common(sym, w, 0);
}

int ats_ilu_setup_async(const IluSymbolic *sym, AtsIluWorkspace *w)
{
    return ats_ilu_setup_common(sym, w, 1);
}

void ats_ilu_free(AtsIluWorkspace *w)
{
    if (!w) {
        return;
    }
    csc_free(w->L);
    csc_free(w->U);
    free(w->Ltmp);
    free(w->Utmp);
    free(w->D);
    memset(w, 0, sizeof(*w));
}

void ats_ilu_factor_sync_mode(AtsIluWorkspace *w, int sweeps, AtsScaleMode mode)
{
    CscMatrix * restrict L = w->L;
    CscMatrix * restrict U = w->U;
    const int n = L->ncols;
    const int * restrict Lp = L->colptr;
    const int * restrict Li = L->rowind;
    const int * restrict Up = U->colptr;
    const int * restrict Ui = U->rowind;
    double * restrict Lx = L->x;
    double * restrict Ux = U->x;
    const double * restrict Lsrc = w->Lsrc;
    const double * restrict Usrc = w->Usrc;
    double * restrict D = w->D;

#pragma omp parallel
    {
        for (int s = 0; s < sweeps; ++s) {
            if (mode == ATS_SCALE_MATERIALIZE) {
#pragma omp for schedule(dynamic, 64)
                for (int i = 0; i < n; ++i) {
                    triangular_solve_row_rhs_onfly(
                        i, Lp, Li, Lsrc, Lx, Up, Ui, Ux);
                    D[i] = 1.0 / Lx[Lp[i + 1] - 1];
                }

#pragma omp for schedule(static)
                for (int j = 0; j < n; ++j) {
                    for (int idx = Lp[j]; idx < Lp[j + 1]; ++idx) {
                        const int i = Li[idx];
                        Lx[idx] *= D[i];
                    }
                }

#pragma omp for schedule(dynamic, 64)
                for (int i = 0; i < n; ++i) {
                    triangular_solve_row_rhs_onfly(
                        i, Up, Ui, Usrc, Ux, Lp, Li, Lx);
                }
            } else {
#pragma omp for schedule(dynamic, 64)
                for (int i = 0; i < n; ++i) {
                    triangular_solve_row_rhs_onfly(
                        i, Lp, Li, Lsrc, Lx, Up, Ui, Ux);
                    D[i] = 1.0 / Lx[Lp[i + 1] - 1];
                }

#pragma omp for schedule(dynamic, 64)
                for (int i = 0; i < n; ++i) {
                    triangular_solve_row_rhs_scaled_unitdiag_onfly(
                        i, Up, Ui, Usrc, Ux, Lp, Li, Lx, D);
                }

                if (s + 1 == sweeps) {
#pragma omp for schedule(static)
                    for (int j = 0; j < n; ++j) {
                        for (int idx = Lp[j]; idx < Lp[j + 1]; ++idx) {
                            const int i = Li[idx];
                            Lx[idx] *= D[i];
                        }
                    }
                }
            }
        }
    }
}

void ats_ilu_factor_sync(AtsIluWorkspace *w, int sweeps)
{
    ats_ilu_factor_sync_mode(w, sweeps, ATS_SCALE_FOLDED);
}

void ats_ilu_factor_sync_sweep_mode(AtsIluWorkspace *w, AtsScaleMode mode)
{
    ats_ilu_factor_sync_mode(w, 1, mode);
}

void ats_ilu_factor_async_begin(AtsIluWorkspace *w)
{
    CscMatrix * L = w->L;
    const int n = L->ncols;
    const int * Lp = L->colptr;
    const int * Li = L->rowind;
    double * Lx = L->x;
    const double * Lsrc = w->Lsrc;
    double * D = w->D;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            D[i] = 1.0 / Lsrc[Lp[i + 1] - 1];
        }

#pragma omp for schedule(static)
        for (int j = 0; j < n; ++j) {
            for (int idx = Lp[j]; idx < Lp[j + 1]; ++idx) {
                const int i = Li[idx];
                Lx[idx] *= D[i];
            }
        }
    }
}

void ats_ilu_factor_async_sweep(AtsIluWorkspace *w)
{
    CscMatrix * L = w->L;
    CscMatrix * U = w->U;
    const int n = L->ncols;
    const int * Lp = L->colptr;
    const int * Li = L->rowind;
    const int * Up = U->colptr;
    const int * Ui = U->rowind;
    double * Lx = L->x;
    double * Ux = U->x;
    double * Ltmp = w->Ltmp;
    double * Utmp = w->Utmp;
    const double * Lsrc = w->Lsrc;
    const double * Usrc = w->Usrc;

#pragma omp parallel for ATS_ASYNC_OMP_FOR_SCHEDULE
    for (int i = 0; i < n; ++i) {
        const int ls = Lp[i];
        const int le = Lp[i + 1];
        const int us = Up[i];
        const int ue = Up[i + 1];

        triangular_solve_row_rhs_unitdiag_onfly_norestrict(
            i, Lp, Li, Lsrc, Ltmp, Up, Ui, Ux);
        async_publish_factor_row(Lx + ls, Ltmp + ls, le - ls);

        triangular_solve_row_rhs_onfly_norestrict(
            i, Up, Ui, Usrc, Utmp, Lp, Li, Lx);
        async_publish_factor_row(Ux + us, Utmp + us, ue - us);
    }
}

void ats_ilu_factor_async(AtsIluWorkspace *w, int sweeps)
{
    ats_ilu_factor_async_begin(w);
    for (int s = 0; s < sweeps; ++s) {
        ats_ilu_factor_async_sweep(w);
    }
}

static int same_pattern(const CscMatrix *A, const CscMatrix *B)
{
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->nnz != B->nnz) {
        return 0;
    }
    if (memcmp(A->colptr, B->colptr,
               ((size_t)A->ncols + 1u) * sizeof(int)) != 0) {
        return 0;
    }
    return memcmp(A->rowind, B->rowind, (size_t)A->nnz * sizeof(int)) == 0;
}

static int ats_ic_setup_common(
    const IluSymbolic *sym,
    AtsIcWorkspace *w,
    double diagonal_shift,
    int allocate_tmp)
{
    memset(w, 0, sizeof(*w));
    if (!(diagonal_shift >= 0.0) || !isfinite(diagonal_shift)) {
        fprintf(stderr, "ATS-IC diagonal shift must be finite and nonnegative\n");
        return -1;
    }
    if (!same_pattern(sym->Lsym, sym->Usym)) {
        fprintf(stderr, "ATS-IC requires matching symmetric factor patterns\n");
        return -1;
    }

    const CscMatrix *Lsrc = sym->Lsym;
    const int n = Lsrc->ncols;
    for (int i = 0; i < n; ++i) {
        const int diagonal = Lsrc->colptr[i + 1] - 1;
        const double pivot = Lsrc->x[diagonal];
        if (diagonal < Lsrc->colptr[i] || Lsrc->rowind[diagonal] != i ||
            !(pivot > 0.0) || !isfinite(pivot)) {
            fprintf(stderr, "ATS-IC requires a positive explicit diagonal at row %d\n", i);
            return -1;
        }
    }

    w->Lsrc_mat = Lsrc;
    w->Lsrc = Lsrc->x;
    w->L = csc_clone(Lsrc);
    if (allocate_tmp) {
        w->Ltmp = (double *)ilu_xmalloc((size_t)Lsrc->nnz * sizeof(double));
    }
    w->diagonal_shift = diagonal_shift;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int diagonal = w->L->colptr[i + 1] - 1;
        w->L->x[diagonal] =
            sqrt((1.0 + diagonal_shift) * w->Lsrc[diagonal]);
    }
    return 0;
}

int ats_ic_setup_async(
    const IluSymbolic *sym,
    AtsIcWorkspace *w,
    double diagonal_shift)
{
    return ats_ic_setup_common(sym, w, diagonal_shift, 1);
}

int ats_ic_setup_sequential(
    const IluSymbolic *sym,
    AtsIcWorkspace *w,
    double diagonal_shift)
{
    return ats_ic_setup_common(sym, w, diagonal_shift, 0);
}

void ats_ic_free(AtsIcWorkspace *w)
{
    if (!w) {
        return;
    }
    csc_free(w->L);
    free(w->Ltmp);
    memset(w, 0, sizeof(*w));
}

int ats_ic_factor_sequential(AtsIcWorkspace *w)
{
    CscMatrix *L = w->L;
    const int n = L->ncols;
    const int *Lp = L->colptr;
    const int *Li = L->rowind;
    double *Lx = L->x;
    const double *Lsrc = w->Lsrc;

    for (int i = 0; i < n; ++i) {
        const int row_start = Lp[i];
        const int diagonal = Lp[i + 1] - 1;
        for (int idx = row_start; idx < diagonal; ++idx) {
            const int j = Li[idx];
            const int other_diagonal = Lp[j + 1] - 1;
            const double dot = intersection_dot_onfly(
                row_start, idx, Li, Lx,
                Lp[j], other_diagonal, Li, Lx);
            const double diag_inv = 1.0 / Lx[other_diagonal];
            Lx[idx] = (Lsrc[idx] - dot) * diag_inv;
        }

        double pivot = (1.0 + w->diagonal_shift) * Lsrc[diagonal];
        for (int idx = row_start; idx < diagonal; ++idx) {
            pivot -= Lx[idx] * Lx[idx];
        }
        if (!(pivot > 0.0) || !isfinite(pivot)) {
            Lx[diagonal] = NAN;
            return -1;
        }
        Lx[diagonal] = sqrt(pivot);
    }
    return 0;
}

int ats_ic_factor_async_sweep(AtsIcWorkspace *w)
{
    CscMatrix *L = w->L;
    const int n = L->ncols;
    const int *Lp = L->colptr;
    const int *Li = L->rowind;
    double *Lx = L->x;
    double *Ltmp = w->Ltmp;
    const double *Lsrc = w->Lsrc;
    int breakdown = 0;

#pragma omp parallel for schedule(guided, ATS_IC_OMP_CHUNK) reduction(|:breakdown)
    for (int i = 0; i < n; ++i) {
        const int row_start = Lp[i];
        const int diagonal = Lp[i + 1] - 1;
        for (int idx = row_start; idx < diagonal; ++idx) {
            const int j = Li[idx];
            const int other_diagonal = Lp[j + 1] - 1;
            const double dot = intersection_dot_onfly_async(
                row_start, idx, Li, Ltmp,
                Lp[j], other_diagonal, Li, Lx);
            const double diag_inv =
                1.0 / async_load_factor_value(Lx + other_diagonal);
            Ltmp[idx] = (Lsrc[idx] - dot) * diag_inv;
        }

        double pivot = (1.0 + w->diagonal_shift) * Lsrc[diagonal];
        for (int idx = row_start; idx < diagonal; ++idx) {
            pivot -= Ltmp[idx] * Ltmp[idx];
        }
        if (!(pivot > 0.0) || !isfinite(pivot)) {
            Ltmp[diagonal] = NAN;
            breakdown = 1;
        } else {
            Ltmp[diagonal] = sqrt(pivot);
        }
        async_publish_factor_row(
            Lx + row_start, Ltmp + row_start, diagonal - row_start + 1);
    }
    return breakdown;
}

int ats_ic_factor_async(AtsIcWorkspace *w, int sweeps)
{
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        if (ats_ic_factor_async_sweep(w) != 0) {
            return -1;
        }
    }
    return 0;
}

int paric_setup(const IluSymbolic *sym, ParIcWorkspace *w)
{
    memset(w, 0, sizeof(*w));
    if (!same_pattern(sym->Lsym, sym->Usym)) {
        fprintf(stderr, "ParIC requires matching symmetric factor patterns\n");
        return -1;
    }

    const CscMatrix *Lsrc = sym->Lsym;
    const int n = Lsrc->ncols;
    for (int i = 0; i < n; ++i) {
        const int diagonal = Lsrc->colptr[i + 1] - 1;
        const double pivot = Lsrc->x[diagonal];
        if (diagonal < Lsrc->colptr[i] || Lsrc->rowind[diagonal] != i ||
            !(pivot > 0.0) || !isfinite(pivot)) {
            fprintf(stderr, "ParIC requires a positive explicit diagonal at row %d\n", i);
            return -1;
        }
    }

    w->Lsrc_mat = Lsrc;
    w->Lsrc = Lsrc->x;
    w->L = csc_clone(Lsrc);
    w->entry_row = (int *)ilu_xmalloc((size_t)Lsrc->nnz * sizeof(int));
    for (int i = 0; i < n; ++i) {
        const int diagonal = w->L->colptr[i + 1] - 1;
        w->L->x[diagonal] = sqrt(w->Lsrc[diagonal]);
        for (int idx = w->L->colptr[i]; idx < w->L->colptr[i + 1]; ++idx) {
            w->entry_row[idx] = i;
        }
    }
    return 0;
}

void paric_free(ParIcWorkspace *w)
{
    if (!w) {
        return;
    }
    csc_free(w->L);
    free(w->entry_row);
    memset(w, 0, sizeof(*w));
}

int paric_factor_async_sweep(ParIcWorkspace *w)
{
    CscMatrix *L = w->L;
    const int *Lp = L->colptr;
    const int *Li = L->rowind;
    double *Lx = L->x;
    const double *Lsrc = w->Lsrc;
    const int *entry_row = w->entry_row;
    int breakdown = 0;

#pragma omp parallel for schedule(dynamic, PARIC_OMP_CHUNK) reduction(|:breakdown)
    for (int idx = 0; idx < L->nnz; ++idx) {
        const int i = entry_row[idx];
        const int j = Li[idx];
        double value;
        if (j < i) {
            const int other_diagonal = Lp[j + 1] - 1;
            const double dot = intersection_dot_onfly_all_async(
                Lp[i], idx, Li, Lx,
                Lp[j], other_diagonal, Li, Lx);
            const double diag_inv =
                1.0 / async_load_factor_value(Lx + other_diagonal);
            value = (Lsrc[idx] - dot) * diag_inv;
        } else {
            double pivot = Lsrc[idx];
            for (int p = Lp[i]; p < idx; ++p) {
                const double entry = async_load_factor_value(Lx + p);
                pivot -= entry * entry;
            }
            if (!(pivot > 0.0) || !isfinite(pivot)) {
                value = NAN;
                breakdown = 1;
            } else {
                value = sqrt(pivot);
            }
        }
        __atomic_store(Lx + idx, &value, __ATOMIC_RELAXED);
    }
    return breakdown;
}

int paric_factor_async(ParIcWorkspace *w, int sweeps)
{
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        if (paric_factor_async_sweep(w) != 0) {
            return -1;
        }
    }
    return 0;
}

static int parilu_build_positions(ParIluWorkspace *w)
{
    const int * restrict I = w->I;
    const int * restrict J = w->J;
    const int * restrict Lp = w->L->colptr;
    const int * restrict Li = w->L->rowind;
    const int * restrict Up = w->U->colptr;
    const int * restrict Ui = w->U->rowind;
    const int nwork = w->S->nnz;

    w->out_pos = (int *)ilu_xmalloc((size_t)nwork * sizeof(int));
    w->diag_pos = (int *)ilu_xmalloc((size_t)nwork * sizeof(int));
    int * restrict out_pos = w->out_pos;
    int * restrict diag_pos = w->diag_pos;

    for (int k = 0; k < nwork; ++k) {
        const int i = I[k];
        const int j = J[k];
        const int lstart = Lp[i];
        const int lend = Lp[i + 1];
        const int ustart = Up[j];

        if (i > j) {
            const int diag = Up[j + 1] - 1;
            if (diag < Up[j] || Ui[diag] != j) {
                return -1;
            }
            out_pos[k] = binary_search_index(Li, j, lstart, lend - 1);
            diag_pos[k] = diag;
        } else {
            out_pos[k] = binary_search_index(Ui, i, ustart, Up[j + 1] - 1);
            diag_pos[k] = -1;
        }

        if (out_pos[k] < 0) {
            return -1;
        }
    }

    return 0;
}

int parilu_setup(const IluSymbolic *sym, ParIluWorkspace *w)
{
    memset(w, 0, sizeof(*w));
    w->S = sym->S;
    w->I = sym->S->rowind;
    w->V = sym->S->x;
    w->J = (int *)ilu_xmalloc((size_t)sym->S->nnz * sizeof(int));
    for (int j = 0; j < sym->S->ncols; ++j) {
        for (int p = sym->S->colptr[j]; p < sym->S->colptr[j + 1]; ++p) {
            w->J[p] = j;
        }
    }

    w->L = csc_clone(sym->Lsymm1);
    w->U = csc_clone(sym->Usym);
    w->Ltmp = (double *)ilu_xmalloc((size_t)w->L->nnz * sizeof(double));
    w->Utmp = (double *)ilu_xmalloc((size_t)w->U->nnz * sizeof(double));
    memcpy(w->Ltmp, w->L->x, (size_t)w->L->nnz * sizeof(double));
    memcpy(w->Utmp, w->U->x, (size_t)w->U->nnz * sizeof(double));
    if (parilu_build_positions(w) != 0) {
        parilu_free(w);
        return -1;
    }
    return 0;
}

void parilu_free(ParIluWorkspace *w)
{
    if (!w) {
        return;
    }
    csc_free(w->L);
    csc_free(w->U);
    free(w->Ltmp);
    free(w->Utmp);
    free(w->J);
    free(w->out_pos);
    free(w->diag_pos);
    memset(w, 0, sizeof(*w));
}

static void parilu_update_onfly_sync(
    ParIluWorkspace *w,
    const double * restrict tmpl,
    const double * restrict tmpu)
{
    const int * restrict I = w->I;
    const int * restrict J = w->J;
    const double * restrict V = w->V;
    double * restrict Lv = w->L->x;
    double * restrict Uv = w->U->x;
    const int * restrict out_pos = w->out_pos;
    const int * restrict diag_pos = w->diag_pos;
    const int * restrict Lp = w->L->colptr;
    const int * restrict Li = w->L->rowind;
    const int * restrict Up = w->U->colptr;
    const int * restrict Ui = w->U->rowind;
    const int nwork = w->S->nnz;

#pragma omp parallel for schedule(dynamic, PARILU_OMP_CHUNK)
    for (int k = 0; k < nwork; ++k) {
        const int i = I[k];
        const int j = J[k];
        const double v = V[k];

        const double dot_full = intersection_dot_onfly(
            Lp[i], Lp[i + 1], Li, tmpl,
            Up[j], Up[j + 1] - 1, Ui, tmpu);

        if (i > j) {
            Lv[out_pos[k]] = (v - dot_full) / tmpu[diag_pos[k]];
        } else {
            Uv[out_pos[k]] = v - dot_full;
        }
    }
}

static void parilu_update_onfly_async(ParIluWorkspace *w)
{
    const int * I = w->I;
    const int * J = w->J;
    const double * V = w->V;
    double * Lv = w->L->x;
    double * Uv = w->U->x;
    const int * out_pos = w->out_pos;
    const int * diag_pos = w->diag_pos;
    const int * Lp = w->L->colptr;
    const int * Li = w->L->rowind;
    const int * Up = w->U->colptr;
    const int * Ui = w->U->rowind;
    const int nwork = w->S->nnz;

#pragma omp parallel for schedule(dynamic, PARILU_OMP_CHUNK)
    for (int k = 0; k < nwork; ++k) {
        const int i = I[k];
        const int j = J[k];
        const double v = V[k];

        const double dot_full = intersection_dot_onfly_norestrict(
            Lp[i], Lp[i + 1], Li, Lv,
            Up[j], Up[j + 1] - 1, Ui, Uv);

        if (i > j) {
            Lv[out_pos[k]] = (v - dot_full) / Uv[diag_pos[k]];
        } else {
            Uv[out_pos[k]] = v - dot_full;
        }
    }
}

void parilu_factor_sync(ParIluWorkspace *w, int sweeps)
{
    for (int s = 0; s < sweeps; ++s) {
        parilu_update_onfly_sync(w, w->Ltmp, w->Utmp);
        if (s + 1 < sweeps) {
            memcpy(w->Ltmp, w->L->x, (size_t)w->L->nnz * sizeof(double));
            memcpy(w->Utmp, w->U->x, (size_t)w->U->nnz * sizeof(double));
        }
    }
}

void parilu_factor_sync_sweep(ParIluWorkspace *w)
{
    parilu_update_onfly_sync(w, w->Ltmp, w->Utmp);
    memcpy(w->Ltmp, w->L->x, (size_t)w->L->nnz * sizeof(double));
    memcpy(w->Utmp, w->U->x, (size_t)w->U->nnz * sizeof(double));
}

void parilu_factor_async_sweep(ParIluWorkspace *w)
{
    parilu_update_onfly_async(w);
}

void parilu_factor_async(ParIluWorkspace *w, int sweeps)
{
    for (int s = 0; s < sweeps; ++s) {
        parilu_factor_async_sweep(w);
    }
}
