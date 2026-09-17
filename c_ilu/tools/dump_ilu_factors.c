#include "ilu.h"
#include "preprocess.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define OBJECTIVE_OMP_CHUNK 4096

typedef struct {
    const char *matrix_path;
    const char *out_dir;
    const char *index_csv;
    int generate_n;
    int k;
    int *sweeps_list;
    int sweeps_count;
    int threads;
    int *thread_list;
    int thread_count;
    int sync_threads;
    int run_seq;
    int run_sync;
    int run_async;
    int run_ats_materialize;
    int run_ats_folded;
    int method_seq;
    int method_ats_sync;
    int method_ats_async;
    int method_ats_ic;
    int method_par_ic;
    int method_par_sync;
    int method_par_async;
    int async_repeats;
    IluSymbolicMode symbolic_mode;
    const char *symbolic_mode_name;
    PreprocessMode preprocess_mode;
    const char *request_csv;
    const char *prepared_structure;
    int prepare_only;
    int check_ic_baseline;
} Options;

typedef struct {
    char method[64];
    char variant[64];
    int repeat;
    int sweeps;
    int threads;
} DumpRequest;

typedef struct {
    DumpRequest *rows;
    int count;
} RequestPlan;

typedef struct {
    double phi;
    double residual_norm;
    double relative_residual;
    int finite;
} ObjectiveValue;

typedef struct {
    int *columns;
    int *factor_positions;
    double matrix_squared_norm;
    int finite;
} ObjectiveWorkspace;

typedef struct {
    const char *matrix_name;
    const Options *opt;
    const CscMatrix *Ause;
    const IluSymbolic *sym;
    const PreprocessTimings *pt;
    const IluSymbolicTimings *st;
    const char *structure_path;
    const char *factor_dir;
    const PreprocessTransform *transform;
    FILE *csv;
    int factor_id;
    int structure_written;
    const RequestPlan *requests;
    const ObjectiveWorkspace *objective;
} DumpContext;

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [--matrix file.mtx | --generate n] --out-dir dir\n"
            "          [--index-csv file.csv]\n"
            "          [--k n] [--sweeps n | --sweeps-list list]\n"
            "          [--threads n | --thread-list list] [--sync-threads n]\n"
            "          [--seq baseline|none]\n"
            "          [--variants sync|async|both] [--async-repeats n]\n"
            "          [--methods all|seq|ats_sync|ats_async|ats_ic|par_ic|par_sync|par_async[,..]]\n"
            "          [--ats-scale materialize|folded|both]\n"
            "          [--symbolic serial|levelset]\n"
            "          [--preprocess none|rcm|mc64|mc64-rcm|diag|diag-rcm|left-diag|left-diag-rcm]\n"
            "          [--request-csv missing_factors.csv]\n"
            "          [--prepare-only | --prepared-structure structure.bin]\n"
            "          [--ic-baseline check|assume-valid]\n",
            prog);
}

static void enable_all_methods(Options *opt)
{
    opt->method_seq = 1;
    opt->method_ats_sync = 1;
    opt->method_ats_async = 1;
    opt->method_par_sync = 1;
    opt->method_par_async = 1;
}

static void parse_methods(const char *s, Options *opt)
{
    opt->method_seq = 0;
    opt->method_ats_sync = 0;
    opt->method_ats_async = 0;
    opt->method_ats_ic = 0;
    opt->method_par_ic = 0;
    opt->method_par_sync = 0;
    opt->method_par_async = 0;

    char *copy = (char *)ilu_xmalloc(strlen(s) + 1);
    strcpy(copy, s);
    for (char *tok = strtok(copy, ", \t"); tok; tok = strtok(NULL, ", \t")) {
        if (strcmp(tok, "all") == 0) {
            enable_all_methods(opt);
        } else if (strcmp(tok, "seq") == 0 || strcmp(tok, "sequential") == 0) {
            opt->method_seq = 1;
        } else if (strcmp(tok, "ats") == 0) {
            opt->method_ats_sync = 1;
            opt->method_ats_async = 1;
        } else if (strcmp(tok, "par") == 0 || strcmp(tok, "parilu") == 0) {
            opt->method_par_sync = 1;
            opt->method_par_async = 1;
        } else if (strcmp(tok, "sync") == 0) {
            opt->method_ats_sync = 1;
            opt->method_par_sync = 1;
        } else if (strcmp(tok, "async") == 0) {
            opt->method_ats_async = 1;
            opt->method_par_async = 1;
        } else if (strcmp(tok, "ats_sync") == 0) {
            opt->method_ats_sync = 1;
        } else if (strcmp(tok, "ats_async") == 0) {
            opt->method_ats_async = 1;
        } else if (strcmp(tok, "ats_ic") == 0) {
            opt->method_ats_ic = 1;
        } else if (strcmp(tok, "par_ic") == 0 || strcmp(tok, "paric") == 0) {
            opt->method_par_ic = 1;
        } else if (strcmp(tok, "par_sync") == 0 || strcmp(tok, "parilu_sync") == 0) {
            opt->method_par_sync = 1;
        } else if (strcmp(tok, "par_async") == 0 || strcmp(tok, "parilu_async") == 0) {
            opt->method_par_async = 1;
        } else {
            fprintf(stderr, "invalid methods entry: %s\n", tok);
            free(copy);
            exit(EXIT_FAILURE);
        }
    }
    if (!opt->method_seq && !opt->method_ats_sync && !opt->method_ats_async &&
        !opt->method_ats_ic &&
        !opt->method_par_ic &&
        !opt->method_par_sync && !opt->method_par_async) {
        fprintf(stderr, "empty methods list\n");
        free(copy);
        exit(EXIT_FAILURE);
    }
    free(copy);
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

static int parse_nonnegative_int(const char *s, const char *name)
{
    char *end = NULL;
    const long v = strtol(s, &end, 10);
    if (!end || *end != '\0' || v < 0 || v > 2147483647L) {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static int parse_int_list(const char *s, int allow_zero, int **values_out)
{
    int capacity = 8;
    int count = 0;
    int *values = (int *)ilu_xmalloc((size_t)capacity * sizeof(int));
    const char *p = s;

    while (*p) {
        while (*p == ',' || isspace((unsigned char)*p)) {
            ++p;
        }
        if (!*p) {
            break;
        }

        char *end = NULL;
        const long v = strtol(p, &end, 10);
        if (end == p || v > 2147483647L || (!allow_zero && v <= 0) || (allow_zero && v < 0)) {
            fprintf(stderr, "invalid list: %s\n", s);
            free(values);
            exit(EXIT_FAILURE);
        }

        if (count == capacity) {
            capacity *= 2;
            values = (int *)ilu_xrealloc(values, (size_t)capacity * sizeof(int));
        }
        values[count++] = (int)v;
        p = end;

        if (*p && *p != ',' && !isspace((unsigned char)*p)) {
            fprintf(stderr, "invalid list separator near: %s\n", p);
            free(values);
            exit(EXIT_FAILURE);
        }
    }

    if (count == 0) {
        fprintf(stderr, "empty list\n");
        free(values);
        exit(EXIT_FAILURE);
    }
    *values_out = values;
    return count;
}

static void set_single_int(int value, int **values, int *count)
{
    free(*values);
    *values = (int *)ilu_xmalloc(sizeof(int));
    (*values)[0] = value;
    *count = 1;
}

static Options parse_options(int argc, char **argv)
{
    Options opt = {
        .matrix_path = NULL,
        .out_dir = "factor_dump",
        .index_csv = NULL,
        .generate_n = 64,
        .k = 1,
        .sweeps_list = NULL,
        .sweeps_count = 0,
        .threads = omp_get_max_threads(),
        .thread_list = NULL,
        .thread_count = 0,
        .sync_threads = 0,
        .run_seq = 1,
        .run_sync = 1,
        .run_async = 1,
        .run_ats_materialize = 0,
        .run_ats_folded = 1,
        .method_seq = 1,
        .method_ats_sync = 1,
        .method_ats_async = 1,
        .method_ats_ic = 0,
        .method_par_ic = 0,
        .method_par_sync = 1,
        .method_par_async = 1,
        .async_repeats = 3,
        .symbolic_mode = ILU_SYMBOLIC_LEVELSET,
        .symbolic_mode_name = "levelset",
        .preprocess_mode = PREPROCESS_NONE,
        .request_csv = NULL,
        .prepared_structure = NULL,
        .prepare_only = 0,
        .check_ic_baseline = 1,
    };

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--matrix") == 0 && i + 1 < argc) {
            opt.matrix_path = argv[++i];
            opt.generate_n = 0;
        } else if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            opt.generate_n = parse_int(argv[++i], "generate");
            opt.matrix_path = NULL;
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            opt.out_dir = argv[++i];
        } else if (strcmp(argv[i], "--index-csv") == 0 && i + 1 < argc) {
            opt.index_csv = argv[++i];
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            opt.k = parse_nonnegative_int(argv[++i], "k");
        } else if (strcmp(argv[i], "--sweeps") == 0 && i + 1 < argc) {
            set_single_int(parse_nonnegative_int(argv[++i], "sweeps"),
                           &opt.sweeps_list, &opt.sweeps_count);
        } else if (strcmp(argv[i], "--sweeps-list") == 0 && i + 1 < argc) {
            free(opt.sweeps_list);
            opt.sweeps_count = parse_int_list(argv[++i], 1, &opt.sweeps_list);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            opt.threads = parse_int(argv[++i], "threads");
            set_single_int(opt.threads, &opt.thread_list, &opt.thread_count);
        } else if (strcmp(argv[i], "--thread-list") == 0 && i + 1 < argc) {
            free(opt.thread_list);
            opt.thread_count = parse_int_list(argv[++i], 0, &opt.thread_list);
        } else if (strcmp(argv[i], "--sync-threads") == 0 && i + 1 < argc) {
            opt.sync_threads = parse_int(argv[++i], "sync-threads");
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "baseline") == 0) {
                opt.run_seq = 1;
            } else if (strcmp(v, "none") == 0) {
                opt.run_seq = 0;
            } else {
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--variants") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "sync") == 0) {
                opt.run_sync = 1;
                opt.run_async = 0;
            } else if (strcmp(v, "async") == 0) {
                opt.run_sync = 0;
                opt.run_async = 1;
            } else if (strcmp(v, "both") == 0) {
                opt.run_sync = 1;
                opt.run_async = 1;
            } else {
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--ats-scale") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "materialize") == 0) {
                opt.run_ats_materialize = 1;
                opt.run_ats_folded = 0;
            } else if (strcmp(v, "folded") == 0) {
                opt.run_ats_materialize = 0;
                opt.run_ats_folded = 1;
            } else if (strcmp(v, "both") == 0) {
                opt.run_ats_materialize = 1;
                opt.run_ats_folded = 1;
            } else {
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--methods") == 0 && i + 1 < argc) {
            parse_methods(argv[++i], &opt);
        } else if (strcmp(argv[i], "--async-repeats") == 0 && i + 1 < argc) {
            opt.async_repeats = parse_int(argv[++i], "async-repeats");
        } else if (strcmp(argv[i], "--symbolic") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "serial") == 0) {
                opt.symbolic_mode = ILU_SYMBOLIC_SERIAL;
                opt.symbolic_mode_name = "serial";
            } else if (strcmp(v, "levelset") == 0) {
                opt.symbolic_mode = ILU_SYMBOLIC_LEVELSET;
                opt.symbolic_mode_name = "levelset";
            } else {
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--preprocess") == 0 && i + 1 < argc) {
            if (parse_preprocess_mode(argv[++i], &opt.preprocess_mode) != 0) {
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--request-csv") == 0 && i + 1 < argc) {
            opt.request_csv = argv[++i];
        } else if (strcmp(argv[i], "--prepared-structure") == 0 && i + 1 < argc) {
            opt.prepared_structure = argv[++i];
        } else if (strcmp(argv[i], "--prepare-only") == 0) {
            opt.prepare_only = 1;
        } else if (strcmp(argv[i], "--ic-baseline") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "check") == 0) {
                opt.check_ic_baseline = 1;
            } else if (strcmp(v, "assume-valid") == 0) {
                opt.check_ic_baseline = 0;
            } else {
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (opt.sweeps_count == 0) {
        set_single_int(5, &opt.sweeps_list, &opt.sweeps_count);
    }
    if (opt.thread_count == 0) {
        set_single_int(opt.threads, &opt.thread_list, &opt.thread_count);
    }
    if (opt.sync_threads == 0) {
        opt.sync_threads = opt.thread_list[opt.thread_count - 1];
    }
    if (opt.prepare_only && opt.prepared_structure) {
        fprintf(stderr, "--prepare-only and --prepared-structure are mutually exclusive\n");
        exit(EXIT_FAILURE);
    }
    return opt;
}

static int split_csv_fields(char *line, char **fields, int capacity)
{
    int count = 0;
    char *field = line;
    while (count < capacity) {
        fields[count++] = field;
        char *comma = strchr(field, ',');
        if (!comma) {
            break;
        }
        *comma = '\0';
        field = comma + 1;
    }
    return count;
}

static int csv_column(char **fields, int count, const char *name)
{
    for (int i = 0; i < count; ++i) {
        if (strcmp(fields[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

static int parse_request_integer(const char *text, int allow_zero, int *value)
{
    char *end = NULL;
    const long parsed = strtol(text, &end, 10);
    if (!end || *end != '\0' || parsed > 2147483647L ||
        (allow_zero ? parsed < 0 : parsed <= 0)) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int request_plan_read(const char *path, RequestPlan *plan)
{
    memset(plan, 0, sizeof(*plan));
    if (!path) {
        return 0;
    }
    FILE *file = fopen(path, "r");
    if (!file) {
        perror(path);
        return -1;
    }

    char line[4096];
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "empty factor request CSV: %s\n", path);
        fclose(file);
        return -1;
    }
    line[strcspn(line, "\r\n")] = '\0';
    char *header[32];
    const int header_count = split_csv_fields(line, header, 32);
    const int method_col = csv_column(header, header_count, "method");
    const int variant_col = csv_column(header, header_count, "variant");
    const int repeat_col = csv_column(header, header_count, "repeat");
    const int sweeps_col = csv_column(header, header_count, "sweeps");
    const int threads_col = csv_column(header, header_count, "threads");
    if (method_col < 0 || variant_col < 0 || repeat_col < 0 ||
        sweeps_col < 0 || threads_col < 0) {
        fprintf(stderr, "factor request CSV is missing required columns: %s\n", path);
        fclose(file);
        return -1;
    }

    int capacity = 32;
    plan->rows = (DumpRequest *)ilu_xmalloc((size_t)capacity * sizeof(DumpRequest));
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') {
            continue;
        }
        char *fields[32];
        const int count = split_csv_fields(line, fields, 32);
        if (method_col >= count || variant_col >= count || repeat_col >= count ||
            sweeps_col >= count || threads_col >= count) {
            fprintf(stderr, "malformed factor request row in %s\n", path);
            fclose(file);
            free(plan->rows);
            memset(plan, 0, sizeof(*plan));
            return -1;
        }
        if (plan->count == capacity) {
            capacity *= 2;
            plan->rows = (DumpRequest *)ilu_xrealloc(
                plan->rows, (size_t)capacity * sizeof(DumpRequest));
        }
        DumpRequest *request = &plan->rows[plan->count];
        if (snprintf(request->method, sizeof(request->method), "%s", fields[method_col]) >=
                (int)sizeof(request->method) ||
            snprintf(request->variant, sizeof(request->variant), "%s", fields[variant_col]) >=
                (int)sizeof(request->variant) ||
            parse_request_integer(fields[repeat_col], 0, &request->repeat) != 0 ||
            parse_request_integer(fields[sweeps_col], 1, &request->sweeps) != 0 ||
            parse_request_integer(fields[threads_col], 0, &request->threads) != 0) {
            fprintf(stderr, "invalid factor request row in %s\n", path);
            fclose(file);
            free(plan->rows);
            memset(plan, 0, sizeof(*plan));
            return -1;
        }
        plan->count += 1;
    }
    if (fclose(file) != 0 || plan->count == 0) {
        fprintf(stderr, "factor request CSV has no requests: %s\n", path);
        free(plan->rows);
        memset(plan, 0, sizeof(*plan));
        return -1;
    }
    return 0;
}

static void request_plan_free(RequestPlan *plan)
{
    free(plan->rows);
    memset(plan, 0, sizeof(*plan));
}

static int request_matches(const DumpContext *ctx, const char *method,
                           const char *variant, int repeat, int sweeps, int threads)
{
    if (!ctx->requests) {
        return 1;
    }
    for (int i = 0; i < ctx->requests->count; ++i) {
        const DumpRequest *request = &ctx->requests->rows[i];
        if (strcmp(request->method, method) == 0 &&
            strcmp(request->variant, variant) == 0 &&
            request->repeat == repeat && request->sweeps == sweeps &&
            request->threads == threads) {
            return 1;
        }
    }
    return 0;
}

static int requested_series_sweeps(const DumpContext *ctx, const char *method,
                                   const char *variant, int repeat, int threads,
                                   int fallback)
{
    if (!ctx->requests) {
        return fallback;
    }
    int maximum = -1;
    for (int i = 0; i < ctx->requests->count; ++i) {
        const DumpRequest *request = &ctx->requests->rows[i];
        if (strcmp(request->method, method) == 0 &&
            strcmp(request->variant, variant) == 0 &&
            request->repeat == repeat && request->threads == threads &&
            request->sweeps > maximum) {
            maximum = request->sweeps;
        }
    }
    return maximum;
}

static const char *matrix_name_from_path(const Options *opt)
{
    if (!opt->matrix_path) {
        return "generated_tridiagonal";
    }
    const char *slash = strrchr(opt->matrix_path, '/');
    return slash ? slash + 1 : opt->matrix_path;
}

static void sanitize_name(const char *input, char *output, size_t n)
{
    size_t q = 0;
    for (const char *p = input; *p && q + 1 < n; ++p) {
        const unsigned char c = (unsigned char)*p;
        if (isalnum(c) || c == '-' || c == '_') {
            output[q++] = (char)c;
        } else {
            output[q++] = '_';
        }
    }
    output[q] = '\0';
}

static int mkdir_p(const char *path)
{
    char tmp[PATH_MAX];
    const size_t len = strlen(path);
    if (len == 0 || len >= sizeof(tmp)) {
        return -1;
    }
    memcpy(tmp, path, len + 1);

    for (char *p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int write_bytes(FILE *f, const void *ptr, size_t size, size_t count)
{
    return fwrite(ptr, size, count, f) == count ? 0 : -1;
}

static int write_magic(FILE *f, const char *magic)
{
    char buf[16] = {0};
    strncpy(buf, magic, sizeof(buf) - 1);
    return write_bytes(f, buf, sizeof(buf), 1);
}

static int write_int32(FILE *f, int value)
{
    const int32_t v = (int32_t)value;
    return write_bytes(f, &v, sizeof(v), 1);
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
        read_int32(f, &nnz) != 0 ||
        nrows <= 0 || ncols <= 0 || nnz < 0) {
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
        read_int32(f, &nnz) != 0 ||
        nrows <= 0 || ncols <= 0 || nnz < 0) {
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

static int read_transform(FILE *f, PreprocessTransform *transform)
{
    memset(transform, 0, sizeof(*transform));
    if (read_int32(f, &transform->n) != 0 || transform->n <= 0) {
        return -1;
    }
    const size_t n = (size_t)transform->n;
    transform->row_old_to_new = (int *)ilu_xmalloc(n * sizeof(int));
    transform->col_old_to_new = (int *)ilu_xmalloc(n * sizeof(int));
    transform->row_scale = (double *)ilu_xmalloc(n * sizeof(double));
    transform->col_scale = (double *)ilu_xmalloc(n * sizeof(double));
    if (read_bytes(f, transform->row_old_to_new, sizeof(int), n) != 0 ||
        read_bytes(f, transform->col_old_to_new, sizeof(int), n) != 0 ||
        read_bytes(f, transform->row_scale, sizeof(double), n) != 0 ||
        read_bytes(f, transform->col_scale, sizeof(double), n) != 0) {
        return -1;
    }
    return 0;
}

static void fill_prepared_symbolic_values(
    const CscMatrix *A,
    CscMatrix *Lraw,
    CscMatrix *U)
{
    for (int j = 0; j < A->ncols; ++j) {
        int pa = A->colptr[j];
        const int pa_end = A->colptr[j + 1];
        int pl = Lraw->colptr[j];
        const int pl_end = Lraw->colptr[j + 1];
        int pu = U->colptr[j];
        const int pu_end = U->colptr[j + 1];
        while (pa < pa_end) {
            const int row = A->rowind[pa];
            if (row <= j) {
                while (pu < pu_end && U->rowind[pu] < row) {
                    pu += 1;
                }
                if (pu < pu_end && U->rowind[pu] == row) {
                    U->x[pu] = A->x[pa];
                }
            }
            if (row >= j) {
                while (pl < pl_end && Lraw->rowind[pl] < row) {
                    pl += 1;
                }
                if (pl < pl_end && Lraw->rowind[pl] == row) {
                    Lraw->x[pl] = A->x[pa];
                }
            }
            pa += 1;
        }
    }
}

static int read_prepared_structure(
    const char *path,
    CscMatrix **Ause_out,
    IluSymbolic *sym,
    PreprocessTransform *transform)
{
    *Ause_out = NULL;
    memset(sym, 0, sizeof(*sym));
    memset(transform, 0, sizeof(*transform));
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    int status = 0;
    if (read_magic(f, "CILUSTRUCTv2") != 0 ||
        read_transform(f, transform) != 0 ||
        !(*Ause_out = read_csc_full(f)) ||
        !(sym->S = read_csc_full(f)) ||
        !(sym->Lsym = read_csc_pattern(f)) ||
        !(sym->Usym = read_csc_pattern(f))) {
        status = -1;
    }
    if (fclose(f) != 0) {
        status = -1;
    }
    if (status == 0) {
        CscMatrix *Lraw = csc_transpose_keepzeros(sym->Lsym);
        if (!Lraw) {
            status = -1;
        } else {
            fill_prepared_symbolic_values(*Ause_out, Lraw, sym->Usym);
            CscMatrix *Lfilled = csc_transpose_keepzeros(Lraw);
            csc_free(Lraw);
            if (!Lfilled) {
                status = -1;
            } else {
                csc_free(sym->Lsym);
                sym->Lsym = Lfilled;
                sym->Lsymm1 = csc_triu_strict(sym->Lsym);
            }
        }
    }
    if (status == 0 &&
        (transform->n != (*Ause_out)->nrows ||
         (*Ause_out)->nrows != (*Ause_out)->ncols ||
         !csc_validate_sorted(*Ause_out) ||
         !csc_validate_sorted(sym->S) ||
         !csc_validate_sorted(sym->Lsym) ||
         !csc_validate_sorted(sym->Usym) ||
         !csc_validate_sorted(sym->Lsymm1))) {
        status = -1;
    }
    if (status != 0) {
        fprintf(stderr, "failed to read prepared structure: %s\n", path);
        ilu_symbolic_free(sym);
        preprocess_transform_free(transform);
        csc_free(*Ause_out);
        *Ause_out = NULL;
    }
    return status;
}

static int write_csc_full(FILE *f, const CscMatrix *A)
{
    if (write_int32(f, A->nrows) != 0 ||
        write_int32(f, A->ncols) != 0 ||
        write_int32(f, A->nnz) != 0 ||
        write_bytes(f, A->colptr, sizeof(int), (size_t)A->ncols + 1u) != 0 ||
        write_bytes(f, A->rowind, sizeof(int), (size_t)A->nnz) != 0 ||
        write_bytes(f, A->x, sizeof(double), (size_t)A->nnz) != 0) {
        return -1;
    }
    return 0;
}

static int write_csc_pattern(FILE *f, const CscMatrix *A)
{
    if (write_int32(f, A->nrows) != 0 ||
        write_int32(f, A->ncols) != 0 ||
        write_int32(f, A->nnz) != 0 ||
        write_bytes(f, A->colptr, sizeof(int), (size_t)A->ncols + 1u) != 0 ||
        write_bytes(f, A->rowind, sizeof(int), (size_t)A->nnz) != 0) {
        return -1;
    }
    return 0;
}

static int write_transform(FILE *f, const PreprocessTransform *transform)
{
    if (write_int32(f, transform->n) != 0 ||
        write_bytes(f, transform->row_old_to_new, sizeof(int), (size_t)transform->n) != 0 ||
        write_bytes(f, transform->col_old_to_new, sizeof(int), (size_t)transform->n) != 0 ||
        write_bytes(f, transform->row_scale, sizeof(double), (size_t)transform->n) != 0 ||
        write_bytes(f, transform->col_scale, sizeof(double), (size_t)transform->n) != 0) {
        return -1;
    }
    return 0;
}

static int write_structure_file(
    const char *path,
    const CscMatrix *Ause,
    const IluSymbolic *sym,
    const PreprocessTransform *transform)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    const int status =
        write_magic(f, "CILUSTRUCTv2") != 0 ||
        write_transform(f, transform) != 0 ||
        write_csc_full(f, Ause) != 0 ||
        write_csc_full(f, sym->S) != 0 ||
        write_csc_pattern(f, sym->Lsym) != 0 ||
        write_csc_pattern(f, sym->Usym) != 0;
    if (fclose(f) != 0 || status) {
        fprintf(stderr, "failed to write structure file: %s\n", path);
        return -1;
    }
    return 0;
}

static int structure_preprocessing(const char *path, const Options *opt,
                                   PreprocessTimings *pt, int writing)
{
    char metadata[PATH_MAX];
    if (snprintf(metadata, sizeof(metadata), "%s.preprocessing", path) >= (int)sizeof(metadata))
        return -1;
    FILE *f = fopen(metadata, writing ? "w" : "r");
    if (!f) {
        fprintf(stderr, "Missing/unwritable prepared preprocessing metadata: %s. Regenerate the structure.\n", metadata);
        return -1;
    }
    int requested = pt->requested_mode, effective = pt->effective_mode;
    int k = opt->k, symbolic = opt->symbolic_mode;
    int failed = 0;
    if (writing) {
        failed = fprintf(f, "CILUPREPROCESSv1\n%d %d %d %d\n", requested, effective, k, symbolic) < 0;
    } else {
        failed = fscanf(f, "CILUPREPROCESSv1\n%d %d %d %d", &requested, &effective, &k, &symbolic) != 4;
        failed |= requested != (int)opt->preprocess_mode || k != opt->k || symbolic != (int)opt->symbolic_mode;
        failed |= effective != requested &&
            !(requested == PREPROCESS_MC64 && effective == PREPROCESS_NONE) &&
            !(requested == PREPROCESS_MC64_RCM && effective == PREPROCESS_RCM);
        if (!failed) {
            pt->requested_mode = (PreprocessMode)requested;
            pt->effective_mode = (PreprocessMode)effective;
            if (requested != effective) preprocess_warn_unavailable();
        }
    }
    failed |= fclose(f) != 0;
    if (failed) fprintf(stderr, "Prepared structure preprocessing/fill metadata mismatch: %s\n", metadata);
    return failed ? -1 : 0;
}

static int ensure_structure_file(DumpContext *ctx)
{
    if (ctx->structure_written) {
        return 0;
    }
    if (write_structure_file(
            ctx->structure_path, ctx->Ause, ctx->sym, ctx->transform) != 0) {
        return -1;
    }
    PreprocessTimings metadata = *ctx->pt;
    if (structure_preprocessing(ctx->structure_path, ctx->opt, &metadata, 1) != 0)
        return -1;
    ctx->structure_written = 1;
    return 0;
}

static int write_factor_values(const char *path,
                               const CscMatrix *Lpattern,
                               const CscMatrix *Upattern,
                               const double *Lx,
                               const double *Ux)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    const int status =
        write_magic(f, "CILUVALSv1") != 0 ||
        write_int32(f, Lpattern->ncols) != 0 ||
        write_int32(f, Lpattern->nnz) != 0 ||
        write_int32(f, Upattern->nnz) != 0 ||
        write_bytes(f, Lx, sizeof(double), (size_t)Lpattern->nnz) != 0 ||
        write_bytes(f, Ux, sizeof(double), (size_t)Upattern->nnz) != 0;
    if (fclose(f) != 0 || status) {
        fprintf(stderr, "failed to write factor values: %s\n", path);
        return -1;
    }
    return 0;
}

static int write_symmetric_factor_values(const char *path,
                                         const CscMatrix *Lpattern,
                                         const double *Lx)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    const int status =
        write_magic(f, "CICVALSv1") != 0 ||
        write_int32(f, Lpattern->ncols) != 0 ||
        write_int32(f, Lpattern->nnz) != 0 ||
        write_bytes(f, Lx, sizeof(double), (size_t)Lpattern->nnz) != 0;
    if (fclose(f) != 0 || status) {
        fprintf(stderr, "failed to write symmetric factor values: %s\n", path);
        return -1;
    }
    return 0;
}

static int method_uses_symmetric_factor(const char *method)
{
    return strcmp(method, "sequential_ic") == 0 ||
           strcmp(method, "ats_ic") == 0 ||
           strcmp(method, "par_ic") == 0;
}

static double factor_checksum_values(const double *Lx, int Lnnz, const double *Ux, int Unnz)
{
    double s = 0.0;
    for (int p = 0; p < Lnnz; ++p) {
        s += fabs(Lx[p]);
    }
    for (int p = 0; p < Unnz; ++p) {
        s += fabs(Ux[p]);
    }
    return s;
}

static double dot_lrow_ucol_prefix(const CscMatrix *Ltrans,
                                   const double *Lx,
                                   const CscMatrix *U,
                                   const double *Ux,
                                   int row,
                                   int col,
                                   int lend,
                                   int uend)
{
    int lp = Ltrans->colptr[row];
    int up = U->colptr[col];
    const int *Li = Ltrans->rowind;
    const int *Ui = U->rowind;
    double dot = 0.0;
    while (lp < lend && up < uend) {
        const int li = Li[lp];
        const int ui = Ui[up];
        if (li == ui) {
            dot += Lx[lp] * Ux[up];
            lp += 1;
            up += 1;
        } else if (li < ui) {
            lp += 1;
        } else {
            up += 1;
        }
    }
    return dot;
}

static void objective_workspace_free(ObjectiveWorkspace *w)
{
    if (!w) {
        return;
    }
    free(w->columns);
    free(w->factor_positions);
    memset(w, 0, sizeof(*w));
}

static int sorted_index(const int *indices, int begin, int end, int target)
{
    while (begin < end) {
        const int middle = begin + (end - begin) / 2;
        if (indices[middle] < target) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin;
}

static int objective_workspace_setup(const CscMatrix *S,
                                     const CscMatrix *Ltrans,
                                     const CscMatrix *U,
                                     ObjectiveWorkspace *w)
{
    memset(w, 0, sizeof(*w));
    w->columns = (int *)ilu_xmalloc((size_t)S->nnz * sizeof(int));
    w->factor_positions = (int *)ilu_xmalloc((size_t)S->nnz * sizeof(int));

    double s2 = 0.0;
    int finite = 1;
    int valid = 1;
#pragma omp parallel for schedule(static) reduction(+:s2) reduction(&:finite) reduction(&:valid)
    for (int col = 0; col < S->ncols; ++col) {
        double local_s2 = 0.0;
        int local_finite = 1;
        int local_valid = 1;
        for (int p = S->colptr[col]; p < S->colptr[col + 1]; ++p) {
            const int row = S->rowind[p];
            const double a2 = S->x[p] * S->x[p];
            w->columns[p] = col;
            local_s2 += a2;
            local_finite &= isfinite(a2) && isfinite(local_s2);

            if (row > col) {
                const int factor_position = sorted_index(
                    Ltrans->rowind, Ltrans->colptr[row], Ltrans->colptr[row + 1], col);
                const int diagonal_position = U->colptr[col + 1] - 1;
                w->factor_positions[p] = factor_position;
                local_valid &= factor_position < Ltrans->colptr[row + 1] &&
                    Ltrans->rowind[factor_position] == col &&
                    diagonal_position >= U->colptr[col] &&
                    U->rowind[diagonal_position] == col;
            } else {
                const int factor_position = sorted_index(
                    U->rowind, U->colptr[col], U->colptr[col + 1], row);
                const int diagonal_position = Ltrans->colptr[row + 1] - 1;
                w->factor_positions[p] = factor_position;
                local_valid &= factor_position < U->colptr[col + 1] &&
                    U->rowind[factor_position] == row &&
                    diagonal_position >= Ltrans->colptr[row] &&
                    Ltrans->rowind[diagonal_position] == row;
            }
        }
        s2 += local_s2;
        finite &= local_finite;
        valid &= local_valid;
    }
    w->matrix_squared_norm = s2;
    w->finite = finite && isfinite(s2);
    if (!valid) {
        fprintf(stderr, "objective setup could not map S onto the factor patterns\n");
        objective_workspace_free(w);
        return -1;
    }
    return 0;
}

static ObjectiveValue masked_objective(const CscMatrix *S,
                                       const CscMatrix *Ltrans,
                                       const double *Lx,
                                       const CscMatrix *U,
                                       const double *Ux,
                                       const ObjectiveWorkspace *w)
{
    double r2 = 0.0;
    int finite = 1;
#pragma omp parallel for schedule(dynamic, OBJECTIVE_OMP_CHUNK) \
    reduction(+:r2) reduction(&:finite)
    for (int p = 0; p < S->nnz; ++p) {
        const int row = S->rowind[p];
        const int col = w->columns[p];
        const int factor_position = w->factor_positions[p];
        double lu;
        if (row > col) {
            const int diagonal_position = U->colptr[col + 1] - 1;
            lu = dot_lrow_ucol_prefix(
                Ltrans, Lx, U, Ux, row, col, factor_position, diagonal_position);
            lu += Lx[factor_position] * Ux[diagonal_position];
        } else {
            const int diagonal_position = Ltrans->colptr[row + 1] - 1;
            lu = dot_lrow_ucol_prefix(
                Ltrans, Lx, U, Ux, row, col, diagonal_position, factor_position);
            lu += Lx[diagonal_position] * Ux[factor_position];
        }
        const double r = S->x[p] - lu;
        const double r_squared = r * r;
        r2 += r_squared;
        finite &= isfinite(r_squared);
    }
    const double s2 = w->matrix_squared_norm;
    ObjectiveValue obj = {
        .phi = 0.5 * r2,
        .residual_norm = sqrt(r2),
        .relative_residual = s2 > 0.0 ? sqrt(r2 / s2) : (r2 == 0.0 ? 0.0 : INFINITY),
        .finite = finite && w->finite && isfinite(r2),
    };
    return obj;
}

static double *materialize_unit_lower_values(const CscMatrix *Lpattern, const CscMatrix *Lstrict)
{
    double *Lx = (double *)ilu_xcalloc((size_t)Lpattern->nnz, sizeof(double));
    int ok = 1;
    for (int col = 0; col < Lpattern->ncols; ++col) {
        int ps = Lstrict->colptr[col];
        const int ps_end = Lstrict->colptr[col + 1];
        for (int p = Lpattern->colptr[col]; p < Lpattern->colptr[col + 1]; ++p) {
            const int row = Lpattern->rowind[p];
            if (row == col) {
                Lx[p] = 1.0;
            } else if (ps < ps_end && Lstrict->rowind[ps] == row) {
                Lx[p] = Lstrict->x[ps++];
            } else {
                ok = 0;
            }
        }
        if (ps != ps_end) {
            ok = 0;
        }
    }
    if (!ok) {
        fprintf(stderr, "failed to map strict unit-lower factor onto explicit L pattern\n");
        free(Lx);
        return NULL;
    }
    return Lx;
}

static void print_csv_header(FILE *f)
{
    fprintf(f,
            "matrix,method,variant,repeat,k,sweeps,threads,preprocess,symbolic_mode,"
            "n,nnz,L_nnz,U_nnz,S_nnz,t_preprocess_total,t_symbolic_total,"
            "t_algorithm_setup,t_factor,t_objective,t_dump,objective,"
            "masked_residual_norm,relative_masked_residual,finite,checksum,"
            "structure_path,factor_path,status,preprocess_requested\n");
    fflush(f);
}

static int emit_factor(DumpContext *ctx,
                       const char *method,
                       const char *variant,
                       int repeat,
                       int sweeps,
                       int threads,
                       double setup_time,
                       double factor_time,
                       const double *Lx,
                       const double *Ux,
                       const char *status)
{
    char factor_path[PATH_MAX];
    char safe_matrix[256];
    if (ensure_structure_file(ctx) != 0) {
        return -1;
    }
    sanitize_name(ctx->matrix_name, safe_matrix, sizeof(safe_matrix));
    ctx->factor_id += 1;
    const int nwritten = snprintf(
        factor_path, sizeof(factor_path),
        "%s/%s_k%d_%s_%s_t%d_s%d_r%d_%06d.bin",
        ctx->factor_dir, safe_matrix, ctx->opt->k, method, variant,
        threads, sweeps, repeat, ctx->factor_id);
    if (nwritten < 0 || (size_t)nwritten >= sizeof(factor_path)) {
        fprintf(stderr, "factor path is too long\n");
        return -1;
    }

    fprintf(stderr,
            "FACTOR method=%s variant=%s repeat=%d sweeps=%d threads=%d "
            "stage=objective\n",
            method, variant, repeat, sweeps, threads);
    fflush(stderr);
    const double tobj0 = wall_seconds();
    ObjectiveValue obj = masked_objective(
        ctx->sym->S, ctx->sym->Lsym, Lx, ctx->sym->Usym, Ux, ctx->objective);
    const double t_objective = wall_seconds() - tobj0;
    fprintf(stderr,
            "FACTOR method=%s variant=%s repeat=%d sweeps=%d threads=%d "
            "stage=dump t_objective=%.9e\n",
            method, variant, repeat, sweeps, threads, t_objective);
    fflush(stderr);
    const double tdump0 = wall_seconds();
    const int symmetric_factor = method_uses_symmetric_factor(method);
    const int dump_status = symmetric_factor ?
        write_symmetric_factor_values(factor_path, ctx->sym->Lsym, Lx) :
        write_factor_values(factor_path, ctx->sym->Lsym, ctx->sym->Usym, Lx, Ux);
    const double t_dump = wall_seconds() - tdump0;
    if (dump_status != 0) {
        return -1;
    }

    if (!obj.finite && strcmp(status, "ok") == 0) {
        status = "nonfinite_objective";
    }

    const double checksum = symmetric_factor ?
        factor_checksum_values(Lx, ctx->sym->Lsym->nnz, NULL, 0) :
        factor_checksum_values(Lx, ctx->sym->Lsym->nnz, Ux, ctx->sym->Usym->nnz);
    fprintf(ctx->csv,
            "%s,%s,%s,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,"
            "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.16e,%.16e,%.16e,%d,"
            "%.16e,%s,%s,%s,%s\n",
            ctx->matrix_name, method, variant, repeat, ctx->opt->k, sweeps, threads,
            preprocess_mode_name(ctx->pt->effective_mode), ctx->opt->symbolic_mode_name,
            ctx->Ause->nrows, ctx->Ause->nnz,
            ctx->sym->Lsym->nnz, ctx->sym->Usym->nnz, ctx->sym->S->nnz,
            ctx->pt->total, ctx->st->total, setup_time, factor_time, t_objective, t_dump,
            obj.phi, obj.residual_norm, obj.relative_residual, obj.finite,
            checksum, ctx->structure_path, factor_path, status,
            preprocess_mode_name(ctx->pt->requested_mode));
    fflush(ctx->csv);
    fprintf(stderr,
            "FACTOR method=%s variant=%s repeat=%d sweeps=%d threads=%d "
            "stage=complete t_objective=%.9e t_dump=%.9e status=%s\n",
            method, variant, repeat, sweeps, threads, t_objective, t_dump, status);
    fflush(stderr);
    return 0;
}

static void emit_failure_timed(DumpContext *ctx,
                               const char *method,
                               const char *variant,
                               int repeat,
                               int sweeps,
                               int threads,
                               double setup_time,
                               double factor_time,
                               const char *status)
{
    fprintf(ctx->csv,
            "%s,%s,%s,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,"
            "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.16e,%.16e,%.16e,%d,"
            "%.16e,%s,%s,%s,%s\n",
            ctx->matrix_name, method, variant, repeat, ctx->opt->k, sweeps, threads,
            preprocess_mode_name(ctx->pt->effective_mode), ctx->opt->symbolic_mode_name,
            ctx->Ause->nrows, ctx->Ause->nnz,
            ctx->sym->Lsym->nnz, ctx->sym->Usym->nnz, ctx->sym->S->nnz,
            ctx->pt->total, ctx->st->total, setup_time, factor_time, 0.0, 0.0,
            INFINITY, INFINITY, INFINITY, 0, INFINITY,
            ctx->structure_written ? ctx->structure_path : "", "", status,
            preprocess_mode_name(ctx->pt->requested_mode));
    fflush(ctx->csv);
}

static void emit_failure(DumpContext *ctx,
                         const char *method,
                         const char *variant,
                         int repeat,
                         int sweeps,
                         int threads,
                         const char *status)
{
    emit_failure_timed(
        ctx, method, variant, repeat, sweeps, threads, 0.0, 0.0, status);
}

static int run_seq(DumpContext *ctx)
{
    SeqIluWorkspace w;
    const double ts = wall_seconds();
    if (seq_ilu_setup(ctx->sym, &w) != 0) {
        emit_failure(ctx, "sequential_ilu", "baseline", 1, 1, 1, "setup_failed");
        return 0;
    }
    const double setup_time = wall_seconds() - ts;
    const double tf = wall_seconds();
    seq_ilu_factor(&w);
    const double factor_time = wall_seconds() - tf;
    const int status = emit_factor(ctx, "sequential_ilu", "baseline", 1, 1, 1,
                                   setup_time, factor_time, w.L->x, w.U->x, "ok");
    seq_ilu_free(&w);
    return status;
}

static int run_seq_ic(DumpContext *ctx, int *baseline_ok)
{
    AtsIcWorkspace w;
    *baseline_ok = 0;
    const double ts = wall_seconds();
    if (ats_ic_setup_sequential(ctx->sym, &w, 0.0) != 0) {
        emit_failure(ctx, "sequential_ic", "baseline", 1, 1, 1, "setup_failed");
        return 0;
    }
    const double setup_time = wall_seconds() - ts;
    const double tf = wall_seconds();
    const int breakdown = ats_ic_factor_sequential(&w);
    const double factor_time = wall_seconds() - tf;
    int status = 0;
    if (breakdown) {
        emit_failure_timed(
            ctx, "sequential_ic", "baseline", 1, 1, 1,
            setup_time, factor_time, "nonpositive_pivot");
    } else {
        if (request_matches(ctx, "sequential_ic", "baseline", 1, 1, 1)) {
            status = emit_factor(
                ctx, "sequential_ic", "baseline", 1, 1, 1,
                setup_time, factor_time, w.L->x, w.L->x, "ok");
        }
        *baseline_ok = 1;
    }
    ats_ic_free(&w);
    return status;
}

static int max_requested_sweeps(const Options *opt)
{
    int max_sweeps = 0;
    for (int i = 0; i < opt->sweeps_count; ++i) {
        if (opt->sweeps_list[i] > max_sweeps) {
            max_sweeps = opt->sweeps_list[i];
        }
    }
    return max_sweeps;
}

static int max_async_sweeps(const Options *opt, int threads)
{
    const int requested = max_requested_sweeps(opt);
    return threads == 1 && requested > 1 ? 1 : requested;
}

static int wants_sweep(const Options *opt, int sweeps)
{
    for (int i = 0; i < opt->sweeps_count; ++i) {
        if (opt->sweeps_list[i] == sweeps) {
            return 1;
        }
    }
    return 0;
}

static int factor_requested(const DumpContext *ctx, const char *method,
                            const char *variant, int repeat, int sweeps, int threads)
{
    if (ctx->requests) {
        return request_matches(ctx, method, variant, repeat, sweeps, threads);
    }
    return wants_sweep(ctx->opt, sweeps);
}

static int run_ats_sync_series(DumpContext *ctx, const char *variant, AtsScaleMode mode, int threads)
{
    const int max_sweeps = requested_series_sweeps(
        ctx, "ats_ilu", variant, 1, threads, max_requested_sweeps(ctx->opt));
    if (max_sweeps < 0) {
        return 0;
    }
    AtsIluWorkspace w;
    const double ts = wall_seconds();
    if (ats_ilu_setup(ctx->sym, &w) != 0) {
        emit_failure(ctx, "ats_ilu", variant, 1, 0, threads, "setup_failed");
        return 0;
    }
    const double setup_time = wall_seconds() - ts;
    double factor_time = 0.0;
    int status = 0;
    if (factor_requested(ctx, "ats_ilu", variant, 1, 0, threads)) {
        status = emit_factor(ctx, "ats_ilu", variant, 1, 0, threads,
                             setup_time, factor_time, w.L->x, w.U->x, "ok");
    }
    for (int sweep = 1; status == 0 && sweep <= max_sweeps; ++sweep) {
        const double tf = wall_seconds();
        ats_ilu_factor_sync_sweep_mode(&w, mode);
        factor_time += wall_seconds() - tf;
        if (factor_requested(ctx, "ats_ilu", variant, 1, sweep, threads)) {
            status = emit_factor(ctx, "ats_ilu", variant, 1, sweep, threads,
                                 setup_time, factor_time, w.L->x, w.U->x, "ok");
        }
    }
    ats_ilu_free(&w);
    return status;
}

static int run_ats_async_series(DumpContext *ctx, int threads, int repeat)
{
    const int max_sweeps = requested_series_sweeps(
        ctx, "ats_ilu", "async", repeat, threads,
        max_async_sweeps(ctx->opt, threads));
    if (max_sweeps < 0) {
        return 0;
    }
    AtsIluWorkspace w;
    const double ts = wall_seconds();
    if (ats_ilu_setup_async(ctx->sym, &w) != 0) {
        emit_failure(ctx, "ats_ilu", "async", repeat, 0, threads, "setup_failed");
        return 0;
    }
    const double setup_time = wall_seconds() - ts;
    double factor_time = 0.0;
    int status = 0;
    const double tb = wall_seconds();
    ats_ilu_factor_async_begin(&w);
    factor_time += wall_seconds() - tb;
    if (factor_requested(ctx, "ats_ilu", "async", repeat, 0, threads)) {
        status = emit_factor(ctx, "ats_ilu", "async", repeat, 0, threads,
                             setup_time, factor_time, w.L->x, w.U->x, "ok");
    }
    for (int sweep = 1; status == 0 && sweep <= max_sweeps; ++sweep) {
        const double tf = wall_seconds();
        ats_ilu_factor_async_sweep(&w);
        factor_time += wall_seconds() - tf;
        if (factor_requested(ctx, "ats_ilu", "async", repeat, sweep, threads)) {
            status = emit_factor(ctx, "ats_ilu", "async", repeat, sweep, threads,
                                 setup_time, factor_time, w.L->x, w.U->x, "ok");
        }
    }
    ats_ilu_free(&w);
    return status;
}

static int run_ats_ic_async_series(DumpContext *ctx, int threads, int repeat)
{
    const int max_sweeps = requested_series_sweeps(
        ctx, "ats_ic", "async", repeat, threads,
        max_async_sweeps(ctx->opt, threads));
    if (max_sweeps < 0) {
        return 0;
    }
    AtsIcWorkspace w;
    const double ts = wall_seconds();
    if (ats_ic_setup_async(ctx->sym, &w, 0.0) != 0) {
        emit_failure(ctx, "ats_ic", "async", repeat, 0, threads, "setup_failed");
        return 0;
    }
    const double setup_time = wall_seconds() - ts;
    double factor_time = 0.0;
    int status = 0;
    int breakdown = 0;
    if (factor_requested(ctx, "ats_ic", "async", repeat, 0, threads)) {
        status = emit_factor(ctx, "ats_ic", "async", repeat, 0, threads,
                             setup_time, factor_time, w.L->x, w.L->x, "ok");
    }
    for (int sweep = 1; status == 0 && !breakdown && sweep <= max_sweeps; ++sweep) {
        const double tf = wall_seconds();
        breakdown = ats_ic_factor_async_sweep(&w);
        factor_time += wall_seconds() - tf;
        if (breakdown) {
            emit_failure_timed(
                ctx, "ats_ic", "async", repeat, sweep, threads,
                setup_time, factor_time, "nonpositive_pivot");
        } else if (factor_requested(ctx, "ats_ic", "async", repeat, sweep, threads)) {
            status = emit_factor(
                ctx, "ats_ic", "async", repeat, sweep, threads,
                setup_time, factor_time, w.L->x, w.L->x, "ok");
        }
    }
    ats_ic_free(&w);
    return status;
}

static int run_par_ic_async_series(DumpContext *ctx, int threads, int repeat)
{
    const int max_sweeps = requested_series_sweeps(
        ctx, "par_ic", "async", repeat, threads,
        max_async_sweeps(ctx->opt, threads));
    if (max_sweeps < 0) {
        return 0;
    }
    ParIcWorkspace w;
    const double ts = wall_seconds();
    if (paric_setup(ctx->sym, &w) != 0) {
        emit_failure(ctx, "par_ic", "async", repeat, 0, threads, "setup_failed");
        return 0;
    }
    const double setup_time = wall_seconds() - ts;
    double factor_time = 0.0;
    int status = 0;
    int breakdown = 0;
    if (factor_requested(ctx, "par_ic", "async", repeat, 0, threads)) {
        status = emit_factor(ctx, "par_ic", "async", repeat, 0, threads,
                             setup_time, factor_time, w.L->x, w.L->x, "ok");
    }
    for (int sweep = 1; status == 0 && !breakdown && sweep <= max_sweeps; ++sweep) {
        const double tf = wall_seconds();
        breakdown = paric_factor_async_sweep(&w);
        factor_time += wall_seconds() - tf;
        if (breakdown) {
            emit_failure_timed(
                ctx, "par_ic", "async", repeat, sweep, threads,
                setup_time, factor_time, "nonpositive_pivot");
        } else if (factor_requested(ctx, "par_ic", "async", repeat, sweep, threads)) {
            status = emit_factor(
                ctx, "par_ic", "async", repeat, sweep, threads,
                setup_time, factor_time, w.L->x, w.L->x, "ok");
        }
    }
    paric_free(&w);
    return status;
}

static int run_par_sync_series(DumpContext *ctx, int threads)
{
    const int max_sweeps = requested_series_sweeps(
        ctx, "parilu", "sync", 1, threads, max_requested_sweeps(ctx->opt));
    if (max_sweeps < 0) {
        return 0;
    }
    ParIluWorkspace w;
    const double ts = wall_seconds();
    if (parilu_setup(ctx->sym, &w) != 0) {
        emit_failure(ctx, "parilu", "sync", 1, 0, threads, "setup_failed");
        return 0;
    }
    const double setup_time = wall_seconds() - ts;
    double factor_time = 0.0;
    int status = 0;
    if (factor_requested(ctx, "parilu", "sync", 1, 0, threads)) {
        double *Lx = materialize_unit_lower_values(ctx->sym->Lsym, w.L);
        if (!Lx) {
            status = -1;
        } else {
            status = emit_factor(ctx, "parilu", "sync", 1, 0, threads,
                                 setup_time, factor_time, Lx, w.U->x, "ok");
            free(Lx);
        }
    }
    for (int sweep = 1; status == 0 && sweep <= max_sweeps; ++sweep) {
        const double tf = wall_seconds();
        parilu_factor_sync_sweep(&w);
        factor_time += wall_seconds() - tf;
        if (factor_requested(ctx, "parilu", "sync", 1, sweep, threads)) {
            double *Lx = materialize_unit_lower_values(ctx->sym->Lsym, w.L);
            if (!Lx) {
                status = -1;
            } else {
                status = emit_factor(ctx, "parilu", "sync", 1, sweep, threads,
                                     setup_time, factor_time, Lx, w.U->x, "ok");
                free(Lx);
            }
        }
    }
    parilu_free(&w);
    return status;
}

static int run_par_async_series(DumpContext *ctx, int threads, int repeat)
{
    const int max_sweeps = requested_series_sweeps(
        ctx, "parilu", "async", repeat, threads,
        max_async_sweeps(ctx->opt, threads));
    if (max_sweeps < 0) {
        return 0;
    }
    ParIluWorkspace w;
    const double ts = wall_seconds();
    if (parilu_setup(ctx->sym, &w) != 0) {
        emit_failure(ctx, "parilu", "async", repeat, 0, threads, "setup_failed");
        return 0;
    }
    const double setup_time = wall_seconds() - ts;
    double factor_time = 0.0;
    int status = 0;
    if (factor_requested(ctx, "parilu", "async", repeat, 0, threads)) {
        double *Lx = materialize_unit_lower_values(ctx->sym->Lsym, w.L);
        if (!Lx) {
            status = -1;
        } else {
            status = emit_factor(ctx, "parilu", "async", repeat, 0, threads,
                                 setup_time, factor_time, Lx, w.U->x, "ok");
            free(Lx);
        }
    }
    for (int sweep = 1; status == 0 && sweep <= max_sweeps; ++sweep) {
        const double tf = wall_seconds();
        parilu_factor_async_sweep(&w);
        factor_time += wall_seconds() - tf;
        if (factor_requested(ctx, "parilu", "async", repeat, sweep, threads)) {
            double *Lx = materialize_unit_lower_values(ctx->sym->Lsym, w.L);
            if (!Lx) {
                status = -1;
            } else {
                status = emit_factor(ctx, "parilu", "async", repeat, sweep, threads,
                                     setup_time, factor_time, Lx, w.U->x, "ok");
                free(Lx);
            }
        }
    }
    parilu_free(&w);
    return status;
}

static int run_dump(DumpContext *ctx)
{
    int ic_baseline_ok = 1;
    const int uses_ic = ctx->opt->method_ats_ic || ctx->opt->method_par_ic;
    if (uses_ic && !ctx->opt->check_ic_baseline && ctx->requests &&
        request_matches(ctx, "sequential_ic", "baseline", 1, 1, 1)) {
        fprintf(stderr, "cannot emit sequential IC with --ic-baseline assume-valid\n");
        return -1;
    }
    if (uses_ic && ctx->opt->check_ic_baseline &&
        run_seq_ic(ctx, &ic_baseline_ok) != 0) {
        return -1;
    }

    if (ctx->opt->run_seq && ctx->opt->method_seq &&
        request_matches(ctx, "sequential_ilu", "baseline", 1, 1, 1) &&
        run_seq(ctx) != 0) {
        return -1;
    }

    if (ctx->opt->run_sync) {
        const int threads = ctx->opt->sync_threads;
        omp_set_num_threads(threads);
        if (ctx->opt->method_ats_sync && ctx->opt->run_ats_materialize &&
            run_ats_sync_series(ctx, "sync_materialize", ATS_SCALE_MATERIALIZE, threads) != 0) {
            return -1;
        }
        if (ctx->opt->method_ats_sync && ctx->opt->run_ats_folded &&
            run_ats_sync_series(ctx, "sync_folded", ATS_SCALE_FOLDED, threads) != 0) {
            return -1;
        }
        if (ctx->opt->method_par_sync && run_par_sync_series(ctx, threads) != 0) {
            return -1;
        }
    }

    for (int ti = 0; ti < ctx->opt->thread_count; ++ti) {
        const int threads = ctx->opt->thread_list[ti];
        omp_set_num_threads(threads);
        if (ctx->opt->run_async && ctx->opt->method_ats_async) {
            for (int rep = 1; rep <= ctx->opt->async_repeats; ++rep) {
                if (run_ats_async_series(ctx, threads, rep) != 0) {
                    return -1;
                }
            }
        }
        if (ic_baseline_ok && ctx->opt->run_async && ctx->opt->method_ats_ic) {
            for (int rep = 1; rep <= ctx->opt->async_repeats; ++rep) {
                if (run_ats_ic_async_series(ctx, threads, rep) != 0) {
                    return -1;
                }
            }
        }
        if (ic_baseline_ok && ctx->opt->run_async && ctx->opt->method_par_ic) {
            for (int rep = 1; rep <= ctx->opt->async_repeats; ++rep) {
                if (run_par_ic_async_series(ctx, threads, rep) != 0) {
                    return -1;
                }
            }
        }
        if (ctx->opt->run_async && ctx->opt->method_par_async) {
            for (int rep = 1; rep <= ctx->opt->async_repeats; ++rep) {
                if (run_par_async_series(ctx, threads, rep) != 0) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    Options opt = parse_options(argc, argv);
    RequestPlan requests;
    if (request_plan_read(opt.request_csv, &requests) != 0) {
        free(opt.thread_list);
        free(opt.sweeps_list);
        return EXIT_FAILURE;
    }
    int setup_threads = 1;
    for (int i = 0; i < opt.thread_count; ++i) {
        if (opt.thread_list[i] > setup_threads) {
            setup_threads = opt.thread_list[i];
        }
    }
    if (opt.sync_threads > setup_threads) {
        setup_threads = opt.sync_threads;
    }
    omp_set_num_threads(setup_threads);

    if (mkdir_p(opt.out_dir) != 0) {
        perror(opt.out_dir);
        free(opt.thread_list);
        free(opt.sweeps_list);
        return EXIT_FAILURE;
    }

    char factor_dir[PATH_MAX];
    if (snprintf(factor_dir, sizeof(factor_dir), "%s/factors", opt.out_dir) >= (int)sizeof(factor_dir) ||
        mkdir_p(factor_dir) != 0) {
        fprintf(stderr, "failed to create factor directory\n");
        free(opt.thread_list);
        free(opt.sweeps_list);
        return EXIT_FAILURE;
    }

    char default_index[PATH_MAX];
    if (!opt.index_csv) {
        if (snprintf(default_index, sizeof(default_index), "%s/factor_index.csv", opt.out_dir) >=
            (int)sizeof(default_index)) {
            fprintf(stderr, "index path is too long\n");
            free(opt.thread_list);
            free(opt.sweeps_list);
            return EXIT_FAILURE;
        }
        opt.index_csv = default_index;
    }

    CscMatrix *A = NULL;
    CscMatrix *Apre = NULL;
    CscMatrix *Aprepared = NULL;
    PreprocessTimings pt;
    PreprocessTransform transform;
    IluSymbolic sym;
    IluSymbolicTimings st;
    memset(&pt, 0, sizeof(pt));
    memset(&transform, 0, sizeof(transform));
    memset(&sym, 0, sizeof(sym));
    memset(&st, 0, sizeof(st));
    if (opt.prepared_structure) {
        if (structure_preprocessing(opt.prepared_structure, &opt, &pt, 0) != 0)
            return EXIT_FAILURE;
        if (read_prepared_structure(
                opt.prepared_structure, &Aprepared, &sym, &transform) != 0) {
            free(opt.thread_list);
            free(opt.sweeps_list);
            request_plan_free(&requests);
            return EXIT_FAILURE;
        }
    } else {
        A = opt.matrix_path ? csc_read_matrix_market(opt.matrix_path)
                            : csc_make_tridiagonal(opt.generate_n);
        if (!A || A->nrows != A->ncols) {
            fprintf(stderr, "input matrix must be square\n");
            csc_free(A);
            free(opt.thread_list);
            free(opt.sweeps_list);
            request_plan_free(&requests);
            return EXIT_FAILURE;
        }
        Apre = preprocess_matrix_with_transform(A, opt.preprocess_mode, &pt, &transform);
        if (opt.preprocess_mode != PREPROCESS_NONE && !Apre) {
            csc_free(A);
            free(opt.thread_list);
            free(opt.sweeps_list);
            request_plan_free(&requests);
            return EXIT_FAILURE;
        }
        const CscMatrix *symbolic_input = Apre ? Apre : A;
        if (ilu_build_symbolic_mode(
                symbolic_input, opt.k, opt.symbolic_mode, &sym, &st) != 0) {
            preprocess_transform_free(&transform);
            csc_free(Apre);
            csc_free(A);
            free(opt.thread_list);
            free(opt.sweeps_list);
            request_plan_free(&requests);
            return EXIT_FAILURE;
        }
    }
    const CscMatrix *Ause = Aprepared ? Aprepared : (Apre ? Apre : A);

    char structure_path[PATH_MAX];
    const int structure_chars = opt.prepared_structure ?
        snprintf(structure_path, sizeof(structure_path), "%s", opt.prepared_structure) :
        snprintf(structure_path, sizeof(structure_path), "%s/structure.bin", opt.out_dir);
    if (structure_chars < 0 || structure_chars >= (int)sizeof(structure_path)) {
        fprintf(stderr, "structure path is too long\n");
        ilu_symbolic_free(&sym);
        preprocess_transform_free(&transform);
        csc_free(Aprepared);
        csc_free(Apre);
        csc_free(A);
        free(opt.thread_list);
        free(opt.sweeps_list);
        return EXIT_FAILURE;
    }

    FILE *csv = fopen(opt.index_csv, "w");
    if (!csv) {
        perror(opt.index_csv);
        ilu_symbolic_free(&sym);
        preprocess_transform_free(&transform);
        csc_free(Aprepared);
        csc_free(Apre);
        csc_free(A);
        free(opt.thread_list);
        free(opt.sweeps_list);
        return EXIT_FAILURE;
    }
    print_csv_header(csv);

    ObjectiveWorkspace objective;
    memset(&objective, 0, sizeof(objective));
    if (!opt.prepare_only) {
        fprintf(stderr, "OBJECTIVE stage=setup entries=%d threads=%d\n",
                sym.S->nnz, setup_threads);
        fflush(stderr);
        const double objective_setup_start = wall_seconds();
        if (objective_workspace_setup(sym.S, sym.Lsym, sym.Usym, &objective) != 0) {
            fclose(csv);
            ilu_symbolic_free(&sym);
            preprocess_transform_free(&transform);
            csc_free(Aprepared);
            csc_free(Apre);
            csc_free(A);
            free(opt.thread_list);
            free(opt.sweeps_list);
            request_plan_free(&requests);
            return EXIT_FAILURE;
        }
        fprintf(stderr, "OBJECTIVE stage=ready t_setup=%.9e\n",
                wall_seconds() - objective_setup_start);
        fflush(stderr);
    }

    DumpContext ctx = {
        .matrix_name = matrix_name_from_path(&opt),
        .opt = &opt,
        .Ause = Ause,
        .sym = &sym,
        .pt = &pt,
        .st = &st,
        .structure_path = structure_path,
        .factor_dir = factor_dir,
        .transform = &transform,
        .csv = csv,
        .factor_id = 0,
        .structure_written = opt.prepared_structure != NULL,
        .requests = opt.request_csv ? &requests : NULL,
        .objective = opt.prepare_only ? NULL : &objective,
    };

    const int status = opt.prepare_only ? ensure_structure_file(&ctx) : run_dump(&ctx);
    if (fclose(csv) != 0) {
        fprintf(stderr, "failed to close index CSV: %s\n", opt.index_csv);
    }

    objective_workspace_free(&objective);
    ilu_symbolic_free(&sym);
    preprocess_transform_free(&transform);
    csc_free(Aprepared);
    csc_free(Apre);
    csc_free(A);
    free(opt.thread_list);
    free(opt.sweeps_list);
    request_plan_free(&requests);
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
