#include "csc.h"

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define TWO_PI 6.283185307179586476925286766559

typedef enum {
    SOLVER_GMRES = 0,
    SOLVER_CG = 1,
} SolverKind;

typedef struct {
    const char *matrix_path;
    const char *factor_index_path;
    const char *out_path;
    SolverKind solver;
    int nrhs;
    int restart;
    int maxiter;
    double tol;
    uint64_t seed;
    int workers;
    double memory_budget_gb;
} Options;

typedef struct {
    int n;
    int *row_old_to_new;
    int *col_old_to_new;
    double *row_scale;
    double *col_scale;
} ReplayTransform;

typedef struct {
    ReplayTransform transform;
    CscMatrix *Ahat;
    CscMatrix *S;
    CscMatrix *Lpattern;
    CscMatrix *Upattern;
} StructureDump;

typedef struct {
    char matrix[256];
    char method[64];
    char variant[64];
    int repeat;
    int k;
    int sweeps;
    int threads;
    char structure_path[PATH_MAX];
    char factor_path[PATH_MAX];
    char status[64];
} FactorRow;

typedef struct {
    const CscMatrix *Ltrans;
    const CscMatrix *U;
    CscMatrix *UT;
    const ReplayTransform *transform;
    double *rhs_hat;
    double *tmp;
    double *zhat;
} TransformedPreconditioner;

typedef struct {
    int iterations;
    int converged;
    int nan_solution;
    int nan_residual;
    int breakdown;
    int stagnated;
    double true_rel_residual;
    double least_squares_rel_residual;
    const char *status;
} GmresResult;

typedef struct {
    int iterations;
    int converged;
    int nan_solution;
    int nan_residual;
    int breakdown;
    int stagnated;
    double true_rel_residual;
    double recursive_rel_residual;
    const char *status;
} CgResult;

typedef struct {
    int available;
    GmresResult gmres;
    CgResult cg;
} ReplayResult;

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s --matrix original.mtx --factor-index factor_index.csv [--out file.csv]\n"
            "          [--solver gmres|cg] [--nrhs n] [--restart n]\n"
            "          [--maxiter n] [--tol x] [--seed n]\n"
            "          [--workers n] [--memory-budget-gb x]\n",
            prog);
}

static int parse_int(const char *s, const char *name)
{
    char *end = NULL;
    const long v = strtol(s, &end, 10);
    if (!end || *end != '\0' || v <= 0 || v > 2147483647L) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static double parse_positive_double(const char *s, const char *name)
{
    char *end = NULL;
    const double v = strtod(s, &end);
    if (!end || *end != '\0' || !(v > 0.0) || !isfinite(v)) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static uint64_t parse_seed(const char *s)
{
    char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0') {
        fprintf(stderr, "invalid seed: %s\n", s);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)v;
}

static Options parse_options(int argc, char **argv)
{
    Options opt = {
        .matrix_path = NULL,
        .factor_index_path = NULL,
        .out_path = NULL,
#ifdef DEFAULT_SOLVER_CG
        .solver = SOLVER_CG,
#else
        .solver = SOLVER_GMRES,
#endif
        .nrhs = 1,
        .restart = 50,
        .maxiter = 2000,
        .tol = 1.0e-8,
        .seed = 1337,
        .workers = 12,
        .memory_budget_gb = 0.0,
    };

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--matrix") == 0 && i + 1 < argc) {
            opt.matrix_path = argv[++i];
        } else if (strcmp(argv[i], "--factor-index") == 0 && i + 1 < argc) {
            opt.factor_index_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            opt.out_path = argv[++i];
        } else if (strcmp(argv[i], "--solver") == 0 && i + 1 < argc) {
            const char *solver = argv[++i];
            if (strcmp(solver, "gmres") == 0) {
                opt.solver = SOLVER_GMRES;
            } else if (strcmp(solver, "cg") == 0) {
                opt.solver = SOLVER_CG;
            } else {
                fprintf(stderr, "invalid solver: %s\n", solver);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--nrhs") == 0 && i + 1 < argc) {
            opt.nrhs = parse_int(argv[++i], "nrhs");
        } else if (strcmp(argv[i], "--restart") == 0 && i + 1 < argc) {
            opt.restart = parse_int(argv[++i], "restart");
        } else if (strcmp(argv[i], "--maxiter") == 0 && i + 1 < argc) {
            opt.maxiter = parse_int(argv[++i], "maxiter");
        } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            opt.tol = parse_positive_double(argv[++i], "tol");
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            opt.seed = parse_seed(argv[++i]);
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            opt.workers = parse_int(argv[++i], "workers");
        } else if (strcmp(argv[i], "--memory-budget-gb") == 0 && i + 1 < argc) {
            opt.memory_budget_gb = parse_positive_double(argv[++i], "memory-budget-gb");
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (!opt.matrix_path || !opt.factor_index_path) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (opt.restart > opt.maxiter) {
        opt.restart = opt.maxiter;
    }
    return opt;
}

static uint64_t rng_next(uint64_t *state)
{
    uint64_t x = *state;
    if (x == 0) {
        x = 0x9e3779b97f4a7c15ULL;
    }
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

static double rng_uniform_open(uint64_t *state)
{
    const uint64_t x = rng_next(state);
    return ((double)((x >> 11) + 1) * 0x1.0p-53);
}

static void fill_random_normal(double *x, int n, uint64_t seed)
{
    uint64_t state = seed == 0 ? 1337 : seed;
    for (int i = 0; i < n; i += 2) {
        const double u1 = rng_uniform_open(&state);
        const double u2 = rng_uniform_open(&state);
        const double r = sqrt(-2.0 * log(u1));
        x[i] = r * cos(TWO_PI * u2);
        if (i + 1 < n) {
            x[i + 1] = r * sin(TWO_PI * u2);
        }
    }
}

static double vec_dot(const double *x, const double *y, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

static double vec_norm(const double *x, int n)
{
    return sqrt(vec_dot(x, x, n));
}

static int vec_has_nonfinite(const double *x, int n)
{
    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            return 1;
        }
    }
    return 0;
}

static void csc_matvec(const CscMatrix *A, const double *x, double *y)
{
    memset(y, 0, (size_t)A->nrows * sizeof(double));
    for (int j = 0; j < A->ncols; ++j) {
        const double xj = x[j];
        for (int p = A->colptr[j]; p < A->colptr[j + 1]; ++p) {
            y[A->rowind[p]] += A->x[p] * xj;
        }
    }
}

static int read_bytes(FILE *f, void *ptr, size_t size, size_t count)
{
    return fread(ptr, size, count, f) == count ? 0 : -1;
}

static int read_magic(FILE *f, const char *expected)
{
    char buf[16] = {0};
    if (read_bytes(f, buf, sizeof(buf), 1) != 0) {
        return -1;
    }
    return strncmp(buf, expected, sizeof(buf)) == 0 ? 0 : -1;
}

static int read_int32(FILE *f, int *value)
{
    int32_t v = 0;
    if (read_bytes(f, &v, sizeof(v), 1) != 0) {
        return -1;
    }
    *value = (int)v;
    return 0;
}

static CscMatrix *read_csc_full(FILE *f)
{
    int nrows = 0;
    int ncols = 0;
    int nnz = 0;
    if (read_int32(f, &nrows) != 0 ||
        read_int32(f, &ncols) != 0 ||
        read_int32(f, &nnz) != 0) {
        return NULL;
    }
    CscMatrix *A = csc_alloc(nrows, ncols, nnz);
    if (read_bytes(f, A->colptr, sizeof(int), (size_t)ncols + 1u) != 0 ||
        read_bytes(f, A->rowind, sizeof(int), (size_t)nnz) != 0 ||
        read_bytes(f, A->x, sizeof(double), (size_t)nnz) != 0) {
        csc_free(A);
        return NULL;
    }
    return A;
}

static CscMatrix *read_csc_pattern(FILE *f)
{
    int nrows = 0;
    int ncols = 0;
    int nnz = 0;
    if (read_int32(f, &nrows) != 0 ||
        read_int32(f, &ncols) != 0 ||
        read_int32(f, &nnz) != 0) {
        return NULL;
    }
    CscMatrix *A = csc_alloc(nrows, ncols, nnz);
    if (read_bytes(f, A->colptr, sizeof(int), (size_t)ncols + 1u) != 0 ||
        read_bytes(f, A->rowind, sizeof(int), (size_t)nnz) != 0) {
        csc_free(A);
        return NULL;
    }
    csc_zero(A);
    return A;
}

static void structure_free(StructureDump *s)
{
    if (!s) {
        return;
    }
    free(s->transform.row_old_to_new);
    free(s->transform.col_old_to_new);
    free(s->transform.row_scale);
    free(s->transform.col_scale);
    csc_free(s->Ahat);
    csc_free(s->S);
    csc_free(s->Lpattern);
    csc_free(s->Upattern);
    memset(s, 0, sizeof(*s));
}

static int read_transform(FILE *f, ReplayTransform *t)
{
    memset(t, 0, sizeof(*t));
    if (read_int32(f, &t->n) != 0) {
        return -1;
    }
    t->row_old_to_new = (int *)ilu_xmalloc((size_t)t->n * sizeof(int));
    t->col_old_to_new = (int *)ilu_xmalloc((size_t)t->n * sizeof(int));
    t->row_scale = (double *)ilu_xmalloc((size_t)t->n * sizeof(double));
    t->col_scale = (double *)ilu_xmalloc((size_t)t->n * sizeof(double));
    if (read_bytes(f, t->row_old_to_new, sizeof(int), (size_t)t->n) != 0 ||
        read_bytes(f, t->col_old_to_new, sizeof(int), (size_t)t->n) != 0 ||
        read_bytes(f, t->row_scale, sizeof(double), (size_t)t->n) != 0 ||
        read_bytes(f, t->col_scale, sizeof(double), (size_t)t->n) != 0) {
        return -1;
    }
    return 0;
}

static int structure_read(const char *path, StructureDump *s)
{
    memset(s, 0, sizeof(*s));
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    int status = 0;
    if (read_magic(f, "CILUSTRUCTv2") != 0 ||
        read_transform(f, &s->transform) != 0 ||
        !(s->Ahat = read_csc_full(f)) ||
        !(s->S = read_csc_full(f)) ||
        !(s->Lpattern = read_csc_pattern(f)) ||
        !(s->Upattern = read_csc_pattern(f))) {
        status = -1;
    }
    if (fclose(f) != 0) {
        status = -1;
    }
    if (status != 0) {
        fprintf(stderr, "failed to read structure dump: %s\n", path);
        structure_free(s);
    }
    return status;
}

static int read_factor_values(const char *path, const CscMatrix *Lpattern,
                              const CscMatrix *Upattern, CscMatrix **Lout,
                              CscMatrix **Uout, int *symmetric_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    int n = 0;
    int Lnnz = 0;
    int Unnz = 0;
    int status = 0;
    CscMatrix *L = NULL;
    CscMatrix *U = NULL;
    int symmetric = 0;
    char magic[16] = {0};
    if (read_bytes(f, magic, sizeof(magic), 1) != 0 ||
        (strncmp(magic, "CILUVALSv1", sizeof(magic)) != 0 &&
         strncmp(magic, "CICVALSv1", sizeof(magic)) != 0)) {
        status = -1;
    } else if (strncmp(magic, "CICVALSv1", sizeof(magic)) == 0) {
        symmetric = 1;
        if (read_int32(f, &n) != 0 ||
            read_int32(f, &Lnnz) != 0 ||
            n != Lpattern->ncols || Lnnz != Lpattern->nnz ||
            Lpattern->nrows != Upattern->nrows ||
            Lpattern->ncols != Upattern->ncols ||
            Lpattern->nnz != Upattern->nnz ||
            memcmp(Lpattern->colptr, Upattern->colptr,
                   ((size_t)Lpattern->ncols + 1u) * sizeof(int)) != 0 ||
            memcmp(Lpattern->rowind, Upattern->rowind,
                   (size_t)Lpattern->nnz * sizeof(int)) != 0) {
            status = -1;
        } else {
            L = csc_clone(Lpattern);
            U = csc_clone(Upattern);
            if (read_bytes(f, L->x, sizeof(double), (size_t)Lnnz) != 0) {
                status = -1;
            } else {
                memcpy(U->x, L->x, (size_t)Lnnz * sizeof(double));
            }
        }
    } else {
        if (read_int32(f, &n) != 0 ||
            read_int32(f, &Lnnz) != 0 ||
            read_int32(f, &Unnz) != 0 ||
            n != Lpattern->ncols || Lnnz != Lpattern->nnz || Unnz != Upattern->nnz) {
            status = -1;
        } else {
            L = csc_clone(Lpattern);
            U = csc_clone(Upattern);
            if (read_bytes(f, L->x, sizeof(double), (size_t)Lnnz) != 0 ||
                read_bytes(f, U->x, sizeof(double), (size_t)Unnz) != 0) {
                status = -1;
            }
        }
    }
    if (fclose(f) != 0) {
        status = -1;
    }
    if (status != 0) {
        fprintf(stderr, "failed to read factor values: %s\n", path);
        csc_free(L);
        csc_free(U);
        return -1;
    }
    *Lout = L;
    *Uout = U;
    *symmetric_out = symmetric;
    return 0;
}

static int transform_is_symmetric(const ReplayTransform *transform)
{
    if (memcmp(transform->row_old_to_new, transform->col_old_to_new,
               (size_t)transform->n * sizeof(int)) != 0 ||
        memcmp(transform->row_scale, transform->col_scale,
               (size_t)transform->n * sizeof(double)) != 0) {
        return 0;
    }
    for (int i = 0; i < transform->n; ++i) {
        if (transform->row_old_to_new[i] < 0 ||
            transform->row_old_to_new[i] >= transform->n ||
            !(transform->row_scale[i] > 0.0) ||
            !isfinite(transform->row_scale[i])) {
            return 0;
        }
    }
    return 1;
}

static int symmetric_factor_is_valid(const CscMatrix *Ltrans)
{
    for (int col = 0; col < Ltrans->ncols; ++col) {
        int diagonal_count = 0;
        for (int p = Ltrans->colptr[col]; p < Ltrans->colptr[col + 1]; ++p) {
            const int row = Ltrans->rowind[p];
            if (row > col || !isfinite(Ltrans->x[p])) {
                return 0;
            }
            if (row == col) {
                diagonal_count += 1;
                if (!(Ltrans->x[p] > 0.0)) {
                    return 0;
                }
            }
        }
        if (diagonal_count != 1) {
            return 0;
        }
    }
    return 1;
}

static int lower_transposed_solve(const CscMatrix *Ltrans, const double *b, double *y)
{
    const int n = Ltrans->ncols;
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        double diag = 0.0;
        for (int p = Ltrans->colptr[i]; p < Ltrans->colptr[i + 1]; ++p) {
            const int j = Ltrans->rowind[p];
            if (j < i) {
                sum -= Ltrans->x[p] * y[j];
            } else if (j == i) {
                diag = Ltrans->x[p];
            }
        }
        if (!isfinite(diag) || fabs(diag) <= 0.0) {
            return -1;
        }
        y[i] = sum / diag;
        if (!isfinite(y[i])) {
            return -1;
        }
    }
    return 0;
}

static int upper_solve_from_transpose(const CscMatrix *UT, const double *b, double *x)
{
    const int n = UT->ncols;
    for (int j = n - 1; j >= 0; --j) {
        double sum = b[j];
        double diag = 0.0;
        for (int p = UT->colptr[j]; p < UT->colptr[j + 1]; ++p) {
            const int k = UT->rowind[p];
            if (k > j) {
                sum -= UT->x[p] * x[k];
            } else if (k == j) {
                diag = UT->x[p];
            }
        }
        if (!isfinite(diag) || fabs(diag) <= 0.0) {
            return -1;
        }
        x[j] = sum / diag;
        if (!isfinite(x[j])) {
            return -1;
        }
    }
    return 0;
}

static int preconditioner_init(TransformedPreconditioner *M,
                               const CscMatrix *Ltrans,
                               const CscMatrix *U,
                               const ReplayTransform *transform)
{
    memset(M, 0, sizeof(*M));
    M->Ltrans = Ltrans;
    M->U = U;
    M->transform = transform;
    M->UT = csc_transpose_keepzeros(U);
    M->rhs_hat = (double *)ilu_xmalloc((size_t)U->ncols * sizeof(double));
    M->tmp = (double *)ilu_xmalloc((size_t)U->ncols * sizeof(double));
    M->zhat = (double *)ilu_xmalloc((size_t)U->ncols * sizeof(double));
    return M->UT ? 0 : -1;
}

static void preconditioner_free(TransformedPreconditioner *M)
{
    csc_free(M->UT);
    free(M->rhs_hat);
    free(M->tmp);
    free(M->zhat);
    memset(M, 0, sizeof(*M));
}

static int preconditioner_apply(const TransformedPreconditioner *M, const double *r, double *z)
{
    const int n = M->transform->n;
    for (int i = 0; i < n; ++i) {
        M->rhs_hat[M->transform->row_old_to_new[i]] = M->transform->row_scale[i] * r[i];
    }
    if (lower_transposed_solve(M->Ltrans, M->rhs_hat, M->tmp) != 0 ||
        upper_solve_from_transpose(M->UT, M->tmp, M->zhat) != 0) {
        return -1;
    }
    for (int j = 0; j < n; ++j) {
        z[j] = M->transform->col_scale[j] * M->zhat[M->transform->col_old_to_new[j]];
    }
    return 0;
}

static double true_relative_residual(const CscMatrix *A, const double *b, const double *x,
                                     double normb, double *work)
{
    csc_matvec(A, x, work);
    for (int i = 0; i < A->nrows; ++i) {
        work[i] = b[i] - work[i];
    }
    return vec_norm(work, A->nrows) / normb;
}

static int solve_upper_hessenberg(const double *H, const double *g, int rows, int k, double *y)
{
    for (int i = k - 1; i >= 0; --i) {
        double sum = g[i];
        for (int j = i + 1; j < k; ++j) {
            sum -= H[(size_t)j * (size_t)rows + (size_t)i] * y[j];
        }
        const double diag = H[(size_t)i * (size_t)rows + (size_t)i];
        if (!isfinite(diag) || fabs(diag) <= 1.0e-300) {
            return -1;
        }
        y[i] = sum / diag;
        if (!isfinite(y[i])) {
            return -1;
        }
    }
    return 0;
}

static void build_candidate(const double *xbase, const double *Z, const double *y,
                            int n, int k, double *x)
{
    memcpy(x, xbase, (size_t)n * sizeof(double));
    for (int j = 0; j < k; ++j) {
        const double alpha = y[j];
        const double *zj = Z + (size_t)j * (size_t)n;
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * zj[i];
        }
    }
}

static void gmres_set_status(GmresResult *res, const char *status)
{
    res->status = status;
    res->converged = strcmp(status, "converged") == 0;
    res->nan_solution = strcmp(status, "nan_solution") == 0;
    res->nan_residual = strcmp(status, "nan_residual") == 0;
    res->breakdown = strcmp(status, "breakdown") == 0 ||
                     strcmp(status, "preconditioner_failed") == 0;
}

static GmresResult gmres_solve(const CscMatrix *A, const TransformedPreconditioner *M,
                               const double *b, int restart, int maxiter, double tol)
{
    const int n = A->nrows;
    const int rows = restart + 1;
    GmresResult res = {
        .iterations = 0,
        .converged = 0,
        .nan_solution = 0,
        .nan_residual = 0,
        .breakdown = 0,
        .stagnated = 0,
        .true_rel_residual = INFINITY,
        .least_squares_rel_residual = INFINITY,
        .status = "maxiter",
    };

    double *xbase = (double *)ilu_xcalloc((size_t)n, sizeof(double));
    double *xcand = (double *)ilu_xcalloc((size_t)n, sizeof(double));
    double *r = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    double *w = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    double *Av = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    double *V = (double *)ilu_xmalloc((size_t)rows * (size_t)n * sizeof(double));
    double *Z = (double *)ilu_xmalloc((size_t)restart * (size_t)n * sizeof(double));
    double *H = (double *)ilu_xcalloc((size_t)rows * (size_t)restart, sizeof(double));
    double *cs = (double *)ilu_xmalloc((size_t)restart * sizeof(double));
    double *sn = (double *)ilu_xmalloc((size_t)restart * sizeof(double));
    double *g = (double *)ilu_xmalloc((size_t)rows * sizeof(double));
    double *y = (double *)ilu_xmalloc((size_t)restart * sizeof(double));

    const double normb = vec_norm(b, n);
    if (!(normb > 0.0) || !isfinite(normb)) {
        res.true_rel_residual = 0.0;
        res.least_squares_rel_residual = 0.0;
        gmres_set_status(&res, "converged");
        goto done;
    }

    double previous_cycle_residual = INFINITY;
    int stagnant_cycles = 0;

    while (res.iterations < maxiter) {
        csc_matvec(A, xbase, r);
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - r[i];
        }
        res.true_rel_residual = vec_norm(r, n) / normb;
        if (!isfinite(res.true_rel_residual)) {
            gmres_set_status(&res, "nan_residual");
            goto done;
        }
        if (res.true_rel_residual <= tol) {
            gmres_set_status(&res, "converged");
            goto done;
        }

        const double beta = vec_norm(r, n);
        if (!(beta > 0.0) || !isfinite(beta)) {
            gmres_set_status(&res, "nan_residual");
            goto done;
        }
        res.least_squares_rel_residual = beta / normb;

        memset(H, 0, (size_t)rows * (size_t)restart * sizeof(double));
        memset(g, 0, (size_t)rows * sizeof(double));
        g[0] = beta;
        for (int i = 0; i < n; ++i) {
            V[i] = r[i] / beta;
        }

        int inner_count = 0;
        int cycle_breakdown = 0;
        for (int j = 0; j < restart && res.iterations < maxiter; ++j) {
            const double *vj = V + (size_t)j * (size_t)n;
            double *zj = Z + (size_t)j * (size_t)n;
            if (preconditioner_apply(M, vj, zj) != 0) {
                gmres_set_status(&res, "preconditioner_failed");
                goto done;
            }
            csc_matvec(A, zj, w);
            if (vec_has_nonfinite(w, n)) {
                gmres_set_status(&res, "nan_residual");
                goto done;
            }

            for (int pass = 0; pass < 2; ++pass) {
                for (int i = 0; i <= j; ++i) {
                    double *vi = V + (size_t)i * (size_t)n;
                    const double hij = vec_dot(vi, w, n);
                    H[(size_t)j * (size_t)rows + (size_t)i] += hij;
                    for (int q = 0; q < n; ++q) {
                        w[q] -= hij * vi[q];
                    }
                }
            }

            const double hn = vec_norm(w, n);
            H[(size_t)j * (size_t)rows + (size_t)(j + 1)] = hn;
            if (isfinite(hn) && hn > 1.0e-300) {
                double *vnext = V + (size_t)(j + 1) * (size_t)n;
                for (int i = 0; i < n; ++i) {
                    vnext[i] = w[i] / hn;
                }
            } else {
                cycle_breakdown = 1;
            }

            for (int i = 0; i < j; ++i) {
                const double h0 = H[(size_t)j * (size_t)rows + (size_t)i];
                const double h1 = H[(size_t)j * (size_t)rows + (size_t)(i + 1)];
                H[(size_t)j * (size_t)rows + (size_t)i] = cs[i] * h0 + sn[i] * h1;
                H[(size_t)j * (size_t)rows + (size_t)(i + 1)] = -sn[i] * h0 + cs[i] * h1;
            }

            const double hdiag = H[(size_t)j * (size_t)rows + (size_t)j];
            const double hsub = H[(size_t)j * (size_t)rows + (size_t)(j + 1)];
            const double denom = hypot(hdiag, hsub);
            if (!(denom > 0.0) || !isfinite(denom)) {
                gmres_set_status(&res, "breakdown");
                goto done;
            }
            cs[j] = hdiag / denom;
            sn[j] = hsub / denom;
            H[(size_t)j * (size_t)rows + (size_t)j] = denom;
            H[(size_t)j * (size_t)rows + (size_t)(j + 1)] = 0.0;

            const double gj = g[j];
            g[j] = cs[j] * gj;
            g[j + 1] = -sn[j] * gj;
            res.iterations += 1;
            inner_count = j + 1;
            res.least_squares_rel_residual = fabs(g[j + 1]) / normb;

            if (solve_upper_hessenberg(H, g, rows, inner_count, y) != 0) {
                gmres_set_status(&res, "breakdown");
                goto done;
            }
            build_candidate(xbase, Z, y, n, inner_count, xcand);
            if (vec_has_nonfinite(xcand, n)) {
                gmres_set_status(&res, "nan_solution");
                goto done;
            }

            res.true_rel_residual = true_relative_residual(A, b, xcand, normb, Av);
            if (!isfinite(res.true_rel_residual)) {
                gmres_set_status(&res, "nan_residual");
                goto done;
            }
            if (res.true_rel_residual <= tol) {
                memcpy(xbase, xcand, (size_t)n * sizeof(double));
                gmres_set_status(&res, "converged");
                goto done;
            }
            if (cycle_breakdown) {
                gmres_set_status(&res, "breakdown");
                goto done;
            }
        }

        if (inner_count == 0 || solve_upper_hessenberg(H, g, rows, inner_count, y) != 0) {
            gmres_set_status(&res, "breakdown");
            goto done;
        }
        build_candidate(xbase, Z, y, n, inner_count, xbase);
        if (vec_has_nonfinite(xbase, n)) {
            gmres_set_status(&res, "nan_solution");
            goto done;
        }
        res.true_rel_residual = true_relative_residual(A, b, xbase, normb, Av);
        if (res.true_rel_residual >= previous_cycle_residual * (1.0 - 1.0e-12)) {
            stagnant_cycles += 1;
            if (stagnant_cycles >= 5) {
                res.stagnated = 1;
                gmres_set_status(&res, "stagnated");
                goto done;
            }
        } else {
            stagnant_cycles = 0;
        }
        previous_cycle_residual = res.true_rel_residual;
    }

done:
    free(y);
    free(g);
    free(sn);
    free(cs);
    free(H);
    free(Z);
    free(V);
    free(Av);
    free(w);
    free(r);
    free(xcand);
    free(xbase);
    return res;
}

static void cg_set_status(CgResult *res, const char *status)
{
    res->status = status;
    res->converged = strcmp(status, "converged") == 0;
    res->nan_solution = strcmp(status, "nan_solution") == 0;
    res->nan_residual = strcmp(status, "nan_residual") == 0;
    res->breakdown = strcmp(status, "preconditioner_failed") == 0 ||
                     strcmp(status, "nonpositive_rz") == 0 ||
                     strcmp(status, "nonpositive_pap") == 0;
}

static CgResult cg_solve(const CscMatrix *A, const TransformedPreconditioner *M,
                         const double *b, int maxiter, double tol)
{
    const int n = A->nrows;
    CgResult res = {
        .iterations = 0,
        .converged = 0,
        .nan_solution = 0,
        .nan_residual = 0,
        .breakdown = 0,
        .stagnated = 0,
        .true_rel_residual = INFINITY,
        .recursive_rel_residual = INFINITY,
        .status = "maxiter",
    };

    double *x = (double *)ilu_xcalloc((size_t)n, sizeof(double));
    double *r = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    double *z = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    double *p = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    double *q = (double *)ilu_xmalloc((size_t)n * sizeof(double));
    double *true_work = (double *)ilu_xmalloc((size_t)n * sizeof(double));

    const double normb = vec_norm(b, n);
    if (!(normb > 0.0) || !isfinite(normb)) {
        res.true_rel_residual = 0.0;
        res.recursive_rel_residual = 0.0;
        cg_set_status(&res, "converged");
        goto done;
    }

    memcpy(r, b, (size_t)n * sizeof(double));
    res.true_rel_residual = 1.0;
    res.recursive_rel_residual = 1.0;
    if (preconditioner_apply(M, r, z) != 0) {
        cg_set_status(&res, "preconditioner_failed");
        goto done;
    }
    double rho = vec_dot(r, z, n);
    if (!isfinite(rho)) {
        cg_set_status(&res, "nan_residual");
        goto done;
    }
    if (!(rho > 0.0)) {
        cg_set_status(&res, "nonpositive_rz");
        goto done;
    }
    memcpy(p, z, (size_t)n * sizeof(double));

    double best_residual = res.true_rel_residual;
    int iterations_without_progress = 0;
    while (res.iterations < maxiter) {
        csc_matvec(A, p, q);
        if (vec_has_nonfinite(q, n)) {
            cg_set_status(&res, "nan_residual");
            goto done;
        }
        const double pap = vec_dot(p, q, n);
        if (!isfinite(pap)) {
            cg_set_status(&res, "nan_residual");
            goto done;
        }
        if (!(pap > 0.0)) {
            cg_set_status(&res, "nonpositive_pap");
            goto done;
        }

        const double alpha = rho / pap;
        if (!isfinite(alpha)) {
            cg_set_status(&res, "nan_solution");
            goto done;
        }
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        res.iterations += 1;
        if (vec_has_nonfinite(x, n)) {
            cg_set_status(&res, "nan_solution");
            goto done;
        }
        if (vec_has_nonfinite(r, n)) {
            cg_set_status(&res, "nan_residual");
            goto done;
        }

        res.recursive_rel_residual = vec_norm(r, n) / normb;
        res.true_rel_residual = true_relative_residual(A, b, x, normb, true_work);
        if (!isfinite(res.recursive_rel_residual) ||
            !isfinite(res.true_rel_residual)) {
            cg_set_status(&res, "nan_residual");
            goto done;
        }
        if (res.true_rel_residual <= tol) {
            cg_set_status(&res, "converged");
            goto done;
        }

        if (res.true_rel_residual < best_residual * (1.0 - 1.0e-8)) {
            best_residual = res.true_rel_residual;
            iterations_without_progress = 0;
        } else {
            iterations_without_progress += 1;
            if (iterations_without_progress >= 500) {
                res.stagnated = 1;
                cg_set_status(&res, "stagnated");
                goto done;
            }
        }

        if (preconditioner_apply(M, r, z) != 0) {
            cg_set_status(&res, "preconditioner_failed");
            goto done;
        }
        const double rho_new = vec_dot(r, z, n);
        if (!isfinite(rho_new)) {
            cg_set_status(&res, "nan_residual");
            goto done;
        }
        if (!(rho_new > 0.0)) {
            cg_set_status(&res, "nonpositive_rz");
            goto done;
        }
        const double beta = rho_new / rho;
        if (!isfinite(beta)) {
            cg_set_status(&res, "nan_residual");
            goto done;
        }
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }
        rho = rho_new;
    }

done:
    free(true_work);
    free(q);
    free(p);
    free(z);
    free(r);
    free(x);
    return res;
}

static int split_csv(char *line, char **fields, int max_fields)
{
    int n = 0;
    char *p = line;
    while (n < max_fields) {
        fields[n++] = p;
        char *comma = strchr(p, ',');
        if (!comma) {
            break;
        }
        *comma = '\0';
        p = comma + 1;
    }
    return n;
}

static int find_column(char **fields, int nfields, const char *name)
{
    for (int i = 0; i < nfields; ++i) {
        if (strcmp(fields[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

static int parse_factor_row(char *line, const int *cols, FactorRow *row)
{
    char *fields[64];
    const int n = split_csv(line, fields, 64);
    for (int i = 0; i < 11; ++i) {
        if (cols[i] < 0 || cols[i] >= n) {
            return -1;
        }
    }
    snprintf(row->matrix, sizeof(row->matrix), "%s", fields[cols[0]]);
    snprintf(row->method, sizeof(row->method), "%s", fields[cols[1]]);
    snprintf(row->variant, sizeof(row->variant), "%s", fields[cols[2]]);
    row->repeat = atoi(fields[cols[3]]);
    row->k = atoi(fields[cols[4]]);
    row->sweeps = atoi(fields[cols[5]]);
    row->threads = atoi(fields[cols[6]]);
    snprintf(row->structure_path, sizeof(row->structure_path), "%s", fields[cols[7]]);
    snprintf(row->factor_path, sizeof(row->factor_path), "%s", fields[cols[8]]);
    snprintf(row->status, sizeof(row->status), "%s", fields[cols[9]]);
    return 0;
}

static void print_header(FILE *out, SolverKind solver)
{
    if (solver == SOLVER_CG) {
        fprintf(out,
                "matrix,method,variant,repeat,k,sweeps,threads,solver,rhs_id,nrhs,maxiter,tol,"
                "iterations,converged,status,true_rel_residual,recursive_rel_residual,"
                "nan_solution,nan_residual,breakdown,stagnated,factor_path\n");
    } else {
        fprintf(out,
                "matrix,method,variant,repeat,k,sweeps,threads,rhs_id,nrhs,restart,maxiter,tol,"
                "iterations,converged,status,true_rel_residual,least_squares_rel_residual,"
                "nan_solution,nan_residual,breakdown,stagnated,factor_path\n");
    }
}

static void print_gmres_result(FILE *out, const FactorRow *row, int rhs_id,
                               const Options *opt, const GmresResult *res)
{
    fprintf(out,
            "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.9e,%d,%d,%s,%.16e,%.16e,%d,%d,%d,%d,%s\n",
            row->matrix, row->method, row->variant, row->repeat, row->k, row->sweeps,
            row->threads, rhs_id, opt->nrhs, opt->restart, opt->maxiter, opt->tol,
            res->iterations, res->converged, res->status,
            res->true_rel_residual, res->least_squares_rel_residual,
            res->nan_solution, res->nan_residual, res->breakdown, res->stagnated,
            row->factor_path);
}

static void print_cg_result(FILE *out, const FactorRow *row, int rhs_id,
                            const Options *opt, const CgResult *res)
{
    fprintf(out,
            "%s,%s,%s,%d,%d,%d,%d,cg,%d,%d,%d,%.9e,%d,%d,%s,%.16e,%.16e,%d,%d,%d,%d,%s\n",
            row->matrix, row->method, row->variant, row->repeat, row->k, row->sweeps,
            row->threads, rhs_id, opt->nrhs, opt->maxiter, opt->tol,
            res->iterations, res->converged, res->status,
            res->true_rel_residual, res->recursive_rel_residual,
            res->nan_solution, res->nan_residual, res->breakdown, res->stagnated,
            row->factor_path);
}

static uint64_t csc_storage_bytes(const CscMatrix *A)
{
    if (!A) {
        return 0;
    }
    return (uint64_t)sizeof(*A) +
           ((uint64_t)A->ncols + 1u) * sizeof(int) +
           (uint64_t)A->nnz * (sizeof(int) + sizeof(double));
}

static uint64_t replay_shared_bytes(const CscMatrix *A, const StructureDump *structure)
{
    return csc_storage_bytes(A) +
           (uint64_t)structure->transform.n *
               (2u * sizeof(int) + 2u * sizeof(double)) +
           csc_storage_bytes(structure->Ahat) +
           csc_storage_bytes(structure->S) +
           csc_storage_bytes(structure->Lpattern) +
           csc_storage_bytes(structure->Upattern);
}

static uint64_t replay_worker_bytes(const Options *opt, const StructureDump *structure)
{
    const uint64_t factor_bytes = csc_storage_bytes(structure->Lpattern) +
                                  2u * csc_storage_bytes(structure->Upattern);
    const uint64_t vector_count = opt->solver == SOLVER_GMRES
        ? (uint64_t)(2 * opt->restart + 11)
        : 11u;
    return factor_bytes + vector_count * (uint64_t)structure->transform.n * sizeof(double);
}

static uint64_t available_memory_bytes(void)
{
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }
    return (uint64_t)pages * (uint64_t)page_size;
}

static int replay_worker_count(const Options *opt, const CscMatrix *A,
                               const StructureDump *structure, int factor_count)
{
    int workers = opt->workers < factor_count ? opt->workers : factor_count;
    const uint64_t shared = replay_shared_bytes(A, structure);
    const uint64_t per_worker = replay_worker_bytes(opt, structure);
    const int processors = omp_get_num_procs();
    if (processors > 0 && workers > processors) {
        workers = processors;
    }
    uint64_t memory_limit = 0;
    const char *memory_mode = "none";
    if (opt->memory_budget_gb > 0.0) {
        const double budget = opt->memory_budget_gb * 1024.0 * 1024.0 * 1024.0;
        memory_limit = (uint64_t)(0.95 * budget);
        memory_mode = "explicit";
        int memory_workers = 1;
        if (memory_limit > shared && per_worker > 0) {
            const uint64_t capacity = (memory_limit - shared) / per_worker;
            memory_workers = capacity > (uint64_t)INT32_MAX
                ? INT32_MAX
                : (int)capacity;
            if (memory_workers < 1) {
                memory_workers = 1;
            }
        }
        if (workers > memory_workers) {
            workers = memory_workers;
        }
    } else {
        const uint64_t available = available_memory_bytes();
        if (available > 0 && per_worker > 0) {
            const uint64_t ceiling = 400ULL * 1024ULL * 1024ULL * 1024ULL;
            const uint64_t total_available = available > UINT64_MAX - shared
                ? UINT64_MAX
                : available + shared;
            const uint64_t budget = total_available < ceiling
                ? total_available
                : ceiling;
            memory_limit = (uint64_t)(0.95 * (double)budget);
            memory_mode = "automatic";
            const uint64_t capacity = memory_limit > shared
                ? (memory_limit - shared) / per_worker
                : 0;
            const int memory_workers = capacity > (uint64_t)INT32_MAX
                ? INT32_MAX
                : (int)capacity;
            if (memory_workers > 0 && workers > memory_workers) {
                workers = memory_workers;
            } else if (memory_workers == 0) {
                workers = 1;
            }
        }
    }
    if (workers < 1) {
        workers = 1;
    }
    fprintf(stderr,
            "REPLAY factors=%d workers_requested=%d workers=%d shared_gib=%.2f "
            "per_worker_gib=%.2f memory_limit_gib=%.1f memory_mode=%s\n",
            factor_count, opt->workers, workers,
            (double)shared / (1024.0 * 1024.0 * 1024.0),
            (double)per_worker / (1024.0 * 1024.0 * 1024.0),
            (double)memory_limit / (1024.0 * 1024.0 * 1024.0), memory_mode);
    return workers;
}

static int replay_factor(const Options *opt, const CscMatrix *A,
                         const StructureDump *structure, const FactorRow *row,
                         ReplayResult *results)
{
    CscMatrix *L = NULL;
    CscMatrix *U = NULL;
    int symmetric_factor = 0;
    if (read_factor_values(row->factor_path, structure->Lpattern, structure->Upattern,
                           &L, &U, &symmetric_factor) != 0) {
        return 0;
    }
    if (opt->solver == SOLVER_CG &&
        (!symmetric_factor || !symmetric_factor_is_valid(L))) {
        fprintf(stderr, "CG requires a valid CICVALSv1 factor: %s\n", row->factor_path);
        csc_free(L);
        csc_free(U);
        return 1;
    }

    TransformedPreconditioner M;
    if (preconditioner_init(&M, L, U, &structure->transform) != 0) {
        preconditioner_free(&M);
        csc_free(L);
        csc_free(U);
        return 0;
    }

    double *xtrue = (double *)ilu_xmalloc((size_t)A->ncols * sizeof(double));
    double *b = (double *)ilu_xmalloc((size_t)A->nrows * sizeof(double));
    for (int rhs_id = 1; rhs_id <= opt->nrhs; ++rhs_id) {
        ReplayResult *result = &results[rhs_id - 1];
        fill_random_normal(xtrue, A->ncols,
                           opt->seed + (uint64_t)rhs_id * 1000003ULL);
        csc_matvec(A, xtrue, b);
        if (opt->solver == SOLVER_CG) {
            result->cg = cg_solve(A, &M, b, opt->maxiter, opt->tol);
        } else {
            result->gmres = gmres_solve(
                A, &M, b, opt->restart, opt->maxiter, opt->tol);
        }
        result->available = 1;
    }

    free(b);
    free(xtrue);
    preconditioner_free(&M);
    csc_free(L);
    csc_free(U);
    return 0;
}

static int replay_group(FILE *out, const Options *opt, const CscMatrix *A,
                        const StructureDump *structure, const FactorRow *rows,
                        int count)
{
    const int workers = replay_worker_count(opt, A, structure, count);
    ReplayResult *results = (ReplayResult *)ilu_xcalloc(
        (size_t)count * (size_t)opt->nrhs, sizeof(*results));
    int fatal_error = 0;

#pragma omp parallel for schedule(dynamic, 1) num_threads(workers) reduction(|:fatal_error)
    for (int i = 0; i < count; ++i) {
        fatal_error |= replay_factor(
            opt, A, structure, &rows[i], results + (size_t)i * (size_t)opt->nrhs);
    }

    for (int i = 0; i < count; ++i) {
        for (int rhs_id = 1; rhs_id <= opt->nrhs; ++rhs_id) {
            const ReplayResult *result =
                &results[(size_t)i * (size_t)opt->nrhs + (size_t)(rhs_id - 1)];
            if (!result->available) {
                continue;
            }
            if (opt->solver == SOLVER_CG) {
                print_cg_result(out, &rows[i], rhs_id, opt, &result->cg);
            } else {
                print_gmres_result(out, &rows[i], rhs_id, opt, &result->gmres);
            }
        }
    }
    free(results);
    return fatal_error;
}

int main(int argc, char **argv)
{
    const Options opt = parse_options(argc, argv);
    CscMatrix *A = csc_read_matrix_market(opt.matrix_path);
    if (!A || A->nrows != A->ncols) {
        fprintf(stderr, "original matrix must be square\n");
        csc_free(A);
        return EXIT_FAILURE;
    }

    FILE *index = fopen(opt.factor_index_path, "r");
    if (!index) {
        perror(opt.factor_index_path);
        csc_free(A);
        return EXIT_FAILURE;
    }
    FILE *out = stdout;
    if (opt.out_path) {
        out = fopen(opt.out_path, "w");
        if (!out) {
            perror(opt.out_path);
            fclose(index);
            csc_free(A);
            return EXIT_FAILURE;
        }
    }

    char line[16384];
    if (!fgets(line, sizeof(line), index)) {
        fprintf(stderr, "empty factor index: %s\n", opt.factor_index_path);
        fclose(index);
        if (out != stdout) {
            fclose(out);
        }
        csc_free(A);
        return EXIT_FAILURE;
    }
    line[strcspn(line, "\r\n")] = '\0';
    char *header_fields[64];
    const int nheader = split_csv(line, header_fields, 64);
    int cols[11] = {
        find_column(header_fields, nheader, "matrix"),
        find_column(header_fields, nheader, "method"),
        find_column(header_fields, nheader, "variant"),
        find_column(header_fields, nheader, "repeat"),
        find_column(header_fields, nheader, "k"),
        find_column(header_fields, nheader, "sweeps"),
        find_column(header_fields, nheader, "threads"),
        find_column(header_fields, nheader, "structure_path"),
        find_column(header_fields, nheader, "factor_path"),
        find_column(header_fields, nheader, "status"),
        find_column(header_fields, nheader, "finite"),
    };

    FactorRow *rows = NULL;
    int row_count = 0;
    int row_capacity = 0;
    while (fgets(line, sizeof(line), index)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') {
            continue;
        }
        FactorRow row;
        if (parse_factor_row(line, cols, &row) != 0) {
            fprintf(stderr, "skipping malformed factor index row\n");
            continue;
        }
        if (strcmp(row.status, "ok") != 0 || row.factor_path[0] == '\0') {
            continue;
        }
        if (row_count == row_capacity) {
            row_capacity = row_capacity == 0 ? 64 : 2 * row_capacity;
            rows = (FactorRow *)ilu_xrealloc(
                rows, (size_t)row_capacity * sizeof(*rows));
        }
        rows[row_count++] = row;
    }
    fclose(index);

    print_header(out, opt.solver);
    int fatal_error = 0;
    for (int start = 0; start < row_count;) {
        int end = start + 1;
        while (end < row_count &&
               strcmp(rows[start].structure_path, rows[end].structure_path) == 0) {
            end += 1;
        }

        StructureDump structure;
        if (structure_read(rows[start].structure_path, &structure) != 0) {
            fatal_error = 1;
            break;
        }
        if (structure.transform.n != A->nrows || structure.transform.n != A->ncols) {
            fprintf(stderr, "structure transform size does not match original matrix\n");
            structure_free(&structure);
            fatal_error = 1;
            break;
        }
        if (opt.solver == SOLVER_CG &&
            !transform_is_symmetric(&structure.transform)) {
            fprintf(stderr,
                    "CG requires identical row/column permutations and scaling\n");
            structure_free(&structure);
            fatal_error = 1;
            break;
        }
        fatal_error |= replay_group(
            out, &opt, A, &structure, rows + start, end - start);
        structure_free(&structure);
        start = end;
    }

    free(rows);
    if (out != stdout) {
        fclose(out);
    }
    csc_free(A);
    return fatal_error ? EXIT_FAILURE : EXIT_SUCCESS;
}
