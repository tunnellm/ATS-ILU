#include "ilu.h"
#include "preprocess.h"

#include <omp.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *matrix_path;
    int generate_n;
    int k;
    int sweeps;
    int repeats;
    int numeric_repeats;
    int threads;
    int *thread_list;
    int thread_count;
    int run_seq;
    int run_sync;
    int run_async;
    int run_ats_materialize;
    int run_ats_folded;
    int method_seq;
    int method_ats_ic;
    int method_par_ic;
    int method_ats_sync;
    int method_ats_async;
    int method_par_sync;
    int method_par_async;
    IluSymbolicMode symbolic_mode;
    const char *symbolic_mode_name;
    PreprocessMode preprocess_mode;
} Options;

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [--matrix file.mtx | --generate n] [--k n] [--sweeps n]\n"
            "          [--repeats n] [--numeric-repeats n]\n"
            "          [--threads n | --thread-list list]\n"
            "          [--seq baseline|none]\n"
            "          [--variants sync|async|both]\n"
            "          [--methods all|seq|ats_sync|ats_async|ats_ic|par_ic|par_sync|par_async[,..]]\n"
            "          [--ats-scale materialize|folded|both]\n"
            "          [--symbolic serial|levelset]\n"
            "          [--preprocess none|rcm|mc64|mc64-rcm|diag|diag-rcm|left-diag|left-diag-rcm]\n",
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
    opt->method_ats_ic = 0;
    opt->method_par_ic = 0;
    opt->method_ats_sync = 0;
    opt->method_ats_async = 0;
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
    if (!opt->method_seq && !opt->method_ats_ic && !opt->method_par_ic &&
        !opt->method_ats_sync && !opt->method_ats_async &&
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

static int parse_thread_list(const char *s, int **values_out)
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
        if (end == p || v <= 0 || v > 2147483647L) {
            fprintf(stderr, "invalid thread-list: %s\n", s);
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
            fprintf(stderr, "invalid thread-list separator near: %s\n", p);
            free(values);
            exit(EXIT_FAILURE);
        }
    }

    if (count == 0) {
        fprintf(stderr, "empty thread-list\n");
        free(values);
        exit(EXIT_FAILURE);
    }
    *values_out = values;
    return count;
}

static Options parse_options(int argc, char **argv)
{
    Options opt = {
        .matrix_path = NULL,
        .generate_n = 64,
        .k = 1,
        .sweeps = 5,
        .repeats = 1,
        .numeric_repeats = 1,
        .threads = omp_get_max_threads(),
        .thread_list = NULL,
        .thread_count = 0,
        .run_seq = 1,
        .run_sync = 1,
        .run_async = 0,
        .run_ats_materialize = 0,
        .run_ats_folded = 1,
        .method_seq = 1,
        .method_ats_ic = 0,
        .method_par_ic = 0,
        .method_ats_sync = 1,
        .method_ats_async = 1,
        .method_par_sync = 1,
        .method_par_async = 1,
        .symbolic_mode = ILU_SYMBOLIC_LEVELSET,
        .symbolic_mode_name = "levelset",
        .preprocess_mode = PREPROCESS_NONE,
    };

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--matrix") == 0 && i + 1 < argc) {
            opt.matrix_path = argv[++i];
            opt.generate_n = 0;
        } else if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            opt.generate_n = parse_int(argv[++i], "generate");
            opt.matrix_path = NULL;
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            opt.k = parse_nonnegative_int(argv[++i], "k");
        } else if (strcmp(argv[i], "--sweeps") == 0 && i + 1 < argc) {
            opt.sweeps = parse_int(argv[++i], "sweeps");
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            opt.repeats = parse_int(argv[++i], "repeats");
        } else if (strcmp(argv[i], "--numeric-repeats") == 0 && i + 1 < argc) {
            opt.numeric_repeats = parse_int(argv[++i], "numeric-repeats");
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            opt.threads = parse_int(argv[++i], "threads");
        } else if (strcmp(argv[i], "--thread-list") == 0 && i + 1 < argc) {
            free(opt.thread_list);
            opt.thread_count = parse_thread_list(argv[++i], &opt.thread_list);
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
        } else if (strcmp(argv[i], "--methods") == 0 && i + 1 < argc) {
            parse_methods(argv[++i], &opt);
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
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (opt.thread_count == 0) {
        opt.thread_list = (int *)ilu_xmalloc(sizeof(int));
        opt.thread_list[0] = opt.threads;
        opt.thread_count = 1;
    }
    return opt;
}

static const char *matrix_name(const Options *opt)
{
    if (!opt->matrix_path) {
        return "generated_tridiagonal";
    }
    const char *slash = strrchr(opt->matrix_path, '/');
    return slash ? slash + 1 : opt->matrix_path;
}

static int trace_ats_async_sweep(const AtsIluWorkspace *w, int sweep)
{
    long long nonfinite_l = 0;
    long long nonfinite_u = 0;
    double max_abs_l = 0.0;
    double max_abs_u = 0.0;
    double min_abs_diag_l = INFINITY;
    double min_abs_diag_u = INFINITY;

    for (int p = 0; p < w->L->nnz; ++p) {
        const double value = w->L->x[p];
        if (!isfinite(value)) {
            nonfinite_l += 1;
        } else if (fabs(value) > max_abs_l) {
            max_abs_l = fabs(value);
        }
    }
    for (int p = 0; p < w->U->nnz; ++p) {
        const double value = w->U->x[p];
        if (!isfinite(value)) {
            nonfinite_u += 1;
        } else if (fabs(value) > max_abs_u) {
            max_abs_u = fabs(value);
        }
    }
    for (int i = 0; i < w->L->ncols; ++i) {
        const double ldiag = w->L->x[w->L->colptr[i + 1] - 1];
        const double udiag = w->U->x[w->U->colptr[i + 1] - 1];
        if (isfinite(ldiag) && fabs(ldiag) < min_abs_diag_l) {
            min_abs_diag_l = fabs(ldiag);
        }
        if (isfinite(udiag) && fabs(udiag) < min_abs_diag_u) {
            min_abs_diag_u = fabs(udiag);
        }
    }

    fprintf(stderr,
            "ats_async_trace sweep=%d checksum=%.17g nonfinite_l=%lld nonfinite_u=%lld "
            "max_abs_l=%.17g max_abs_u=%.17g min_abs_diag_l=%.17g min_abs_diag_u=%.17g\n",
            sweep, ilu_workspace_checksum(w->L, w->U),
            nonfinite_l, nonfinite_u,
            max_abs_l, max_abs_u, min_abs_diag_l, min_abs_diag_u);
    fflush(stderr);
    return nonfinite_l != 0 || nonfinite_u != 0;
}

static int effective_async_sweeps(const Options *opt, int threads)
{
    return threads == 1 && opt->sweeps > 1 ? 1 : opt->sweeps;
}

static void print_row(const char *matrix,
                      const char *method,
                      const char *variant,
                      int repeat,
                      int numeric_repeat,
                      int k,
                      int sweeps,
                      int threads,
                      const char *preprocess,
                      const PreprocessTimings *pt,
                      const char *symbolic_mode,
                      const CscMatrix *A,
                      const IluSymbolic *sym,
                      const IluSymbolicTimings *st,
                      double algorithm_setup,
                      double factor_time,
                      double checksum)
{
    const double total = pt->total + st->total + algorithm_setup + factor_time;
    const double preprocess_reorder_apply = pt->mc64_apply + pt->diag_scale + pt->rcm_apply;
    printf("%s,%s,%s,%d,%d,%d,%d,%d,%s,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.16e,%s\n",
           matrix, method, variant, repeat, numeric_repeat, k, sweeps, threads,
           preprocess_mode_name(pt->effective_mode), pt->mc64, pt->diag_scale, pt->rcm, pt->total,
           pt->mc64_prepare, pt->mc64_match, pt->mc64_apply,
           pt->rcm_order, pt->rcm_apply, preprocess_reorder_apply,
           pt->diag_nonzeros_after_mc64, pt->rcm_levels_hint,
           symbolic_mode, st->symbolic_levels, st->symbolic_fallback,
           A->nrows, A->nnz, sym->Lsym->nnz, sym->Usym->nnz, sym->S->nnz,
           st->symbolic_pattern, st->numeric_fill, st->transform,
           algorithm_setup, factor_time, total, checksum, preprocess);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    Options opt = parse_options(argc, argv);
    int setup_threads = opt.thread_list[0];
    for (int i = 1; i < opt.thread_count; ++i) {
        if (opt.thread_list[i] > setup_threads) {
            setup_threads = opt.thread_list[i];
        }
    }
    omp_set_num_threads(setup_threads);

    CscMatrix *A = NULL;
    if (opt.matrix_path) {
        A = csc_read_matrix_market(opt.matrix_path);
    } else {
        A = csc_make_tridiagonal(opt.generate_n);
    }
    if (!A) {
        return EXIT_FAILURE;
    }
    if (A->nrows != A->ncols) {
        fprintf(stderr, "input matrix must be square\n");
        csc_free(A);
        return EXIT_FAILURE;
    }

    printf("matrix,method,variant,repeat,numeric_repeat,k,sweeps,threads,preprocess,"
           "t_preprocess_mc64,t_preprocess_diag,t_preprocess_rcm,t_preprocess_total,"
           "t_mc64_prepare,t_mc64_match,t_mc64_apply,"
           "t_rcm_order,t_rcm_apply,t_preprocess_reorder_apply,"
           "diag_nonzeros_after_mc64,rcm_levels_hint,"
           "symbolic_mode,symbolic_levels,symbolic_fallback,n,nnz,L_nnz,U_nnz,S_nnz,"
           "t_symbolic_pattern,t_numeric_fill,t_transform,t_algorithm_setup,"
           "t_factor,t_total,checksum,preprocess_requested\n");
    fflush(stdout);

    const char *name = matrix_name(&opt);
    const char *preprocess_name = preprocess_mode_name(opt.preprocess_mode);

    for (int rep = 1; rep <= opt.repeats; ++rep) {
        PreprocessTimings pt;
        CscMatrix *Apre = preprocess_matrix(A, opt.preprocess_mode, &pt);
        if (opt.preprocess_mode != PREPROCESS_NONE && !Apre) {
            csc_free(A);
            return EXIT_FAILURE;
        }
        const CscMatrix *Ause = Apre ? Apre : A;

        IluSymbolic sym;
        IluSymbolicTimings st;
        if (ilu_build_symbolic_mode(Ause, opt.k, opt.symbolic_mode, &sym, &st) != 0) {
            csc_free(Apre);
            csc_free(A);
            return EXIT_FAILURE;
        }

        int ic_baseline_ok = 1;
        if (opt.method_ats_ic || opt.method_par_ic) {
            AtsIcWorkspace w;
            omp_set_num_threads(1);
            const double ts = wall_seconds();
            const int setup_status = ats_ic_setup_sequential(&sym, &w, 0.0);
            const double setup_time = wall_seconds() - ts;
            double factor_time = 0.0;
            int breakdown = 0;
            if (setup_status == 0) {
                const double tf = wall_seconds();
                breakdown = ats_ic_factor_sequential(&w);
                factor_time = wall_seconds() - tf;
            }
            ic_baseline_ok = setup_status == 0 && breakdown == 0;
            print_row(name, "sequential_ic",
                      setup_status != 0 ? "baseline_setup_failed" :
                      breakdown != 0 ? "baseline_nonpositive_pivot" : "baseline",
                      rep, 1, opt.k, 1, 1,
                      preprocess_name, &pt,
                      opt.symbolic_mode_name,
                      Ause, &sym, &st, setup_time, factor_time,
                      ic_baseline_ok ? ilu_workspace_checksum(w.L, w.L) : NAN);
            if (setup_status == 0) {
                ats_ic_free(&w);
            }
        }

        if (opt.run_seq && opt.method_seq) {
            omp_set_num_threads(1);
            for (int nrep = 1; nrep <= opt.numeric_repeats; ++nrep) {
                SeqIluWorkspace w;
                const double ts = wall_seconds();
                if (seq_ilu_setup(&sym, &w) != 0) {
                    ilu_symbolic_free(&sym);
                    csc_free(Apre);
                    csc_free(A);
                    return EXIT_FAILURE;
                }
                const double setup_time = wall_seconds() - ts;
                const double tf = wall_seconds();
                seq_ilu_factor(&w);
                const double factor_time = wall_seconds() - tf;
                print_row(name, "sequential_ilu", "baseline", rep, nrep,
                          opt.k, 1, 1,
                          preprocess_name, &pt,
                          opt.symbolic_mode_name,
                          Ause, &sym, &st, setup_time, factor_time,
                          ilu_workspace_checksum(w.L, w.U));
                seq_ilu_free(&w);
            }
        }

        for (int ti = 0; ti < opt.thread_count; ++ti) {
            const int threads = opt.thread_list[ti];
            const int async_sweeps = effective_async_sweeps(&opt, threads);
            omp_set_num_threads(threads);
            for (int nrep = 1; nrep <= opt.numeric_repeats; ++nrep) {
            if (ic_baseline_ok && opt.run_async && opt.method_ats_ic) {
                AtsIcWorkspace w;
                const double ts = wall_seconds();
                const int setup_status = ats_ic_setup_async(&sym, &w, 0.0);
                const double setup_time = wall_seconds() - ts;
                double factor_time = 0.0;
                int breakdown = 0;
                if (setup_status == 0) {
                    const double tf = wall_seconds();
                    breakdown = ats_ic_factor_async(&w, async_sweeps);
                    factor_time = wall_seconds() - tf;
                }
                print_row(name, "ats_ic",
                          setup_status != 0 ? "async_setup_failed" :
                          breakdown != 0 ? "async_nonpositive_pivot" : "async",
                          rep, nrep, opt.k, async_sweeps, threads,
                          preprocess_name, &pt,
                          opt.symbolic_mode_name,
                          Ause, &sym, &st, setup_time, factor_time,
                          setup_status == 0 && breakdown == 0 ?
                              ilu_workspace_checksum(w.L, w.L) : NAN);
                if (setup_status == 0) {
                    ats_ic_free(&w);
                }
            }

            if (ic_baseline_ok && opt.run_async && opt.method_par_ic) {
                ParIcWorkspace w;
                const double ts = wall_seconds();
                const int setup_status = paric_setup(&sym, &w);
                const double setup_time = wall_seconds() - ts;
                double factor_time = 0.0;
                int breakdown = 0;
                if (setup_status == 0) {
                    const double tf = wall_seconds();
                    breakdown = paric_factor_async(&w, async_sweeps);
                    factor_time = wall_seconds() - tf;
                }
                print_row(name, "par_ic",
                          setup_status != 0 ? "async_setup_failed" :
                          breakdown != 0 ? "async_nonpositive_pivot" : "async",
                          rep, nrep, opt.k, async_sweeps, threads,
                          preprocess_name, &pt,
                          opt.symbolic_mode_name,
                          Ause, &sym, &st, setup_time, factor_time,
                          setup_status == 0 && breakdown == 0 ?
                              ilu_workspace_checksum(w.L, w.L) : NAN);
                if (setup_status == 0) {
                    paric_free(&w);
                }
            }

            if (opt.run_sync && opt.method_ats_sync && opt.run_ats_materialize) {
                AtsIluWorkspace w;
                const double ts = wall_seconds();
                if (ats_ilu_setup(&sym, &w) != 0) {
                    ilu_symbolic_free(&sym);
                    csc_free(Apre);
                    csc_free(A);
                    return EXIT_FAILURE;
                }
                const double setup_time = wall_seconds() - ts;
                const double tf = wall_seconds();
                ats_ilu_factor_sync_mode(&w, opt.sweeps, ATS_SCALE_MATERIALIZE);
                const double factor_time = wall_seconds() - tf;
                print_row(name, "ats_ilu", "sync_materialize", rep, nrep,
                          opt.k, opt.sweeps, threads,
                          preprocess_name, &pt,
                          opt.symbolic_mode_name,
                          Ause, &sym, &st, setup_time, factor_time,
                          ilu_workspace_checksum(w.L, w.U));
                ats_ilu_free(&w);
            }

            if (opt.run_sync && opt.method_ats_sync && opt.run_ats_folded) {
                AtsIluWorkspace w;
                const double ts = wall_seconds();
                if (ats_ilu_setup(&sym, &w) != 0) {
                    ilu_symbolic_free(&sym);
                    csc_free(Apre);
                    csc_free(A);
                    return EXIT_FAILURE;
                }
                const double setup_time = wall_seconds() - ts;
                const double tf = wall_seconds();
                ats_ilu_factor_sync_mode(&w, opt.sweeps, ATS_SCALE_FOLDED);
                const double factor_time = wall_seconds() - tf;
                print_row(name, "ats_ilu", "sync_folded", rep, nrep,
                          opt.k, opt.sweeps, threads,
                          preprocess_name, &pt,
                          opt.symbolic_mode_name,
                          Ause, &sym, &st, setup_time, factor_time,
                          ilu_workspace_checksum(w.L, w.U));
                ats_ilu_free(&w);
            }

            if (opt.run_async && opt.method_ats_async) {
                AtsIluWorkspace w;
                const double ts = wall_seconds();
                if (ats_ilu_setup_async(&sym, &w) != 0) {
                    ilu_symbolic_free(&sym);
                    csc_free(Apre);
                    csc_free(A);
                    return EXIT_FAILURE;
                }
                const double setup_time = wall_seconds() - ts;
                const double tf = wall_seconds();
                if (getenv("ATS_ILU_TRACE_ASYNC")) {
                    ats_ilu_factor_async_begin(&w);
                    trace_ats_async_sweep(&w, 0);
                    for (int sweep = 1; sweep <= async_sweeps; ++sweep) {
                        ats_ilu_factor_async_sweep(&w);
                        if (trace_ats_async_sweep(&w, sweep)) {
                            break;
                        }
                    }
                } else {
                    ats_ilu_factor_async(&w, async_sweeps);
                }
                const double factor_time = wall_seconds() - tf;
                print_row(name, "ats_ilu", "async", rep, nrep,
                          opt.k, async_sweeps, threads,
                          preprocess_name, &pt,
                          opt.symbolic_mode_name,
                          Ause, &sym, &st, setup_time, factor_time,
                          ilu_workspace_checksum(w.L, w.U));
                ats_ilu_free(&w);
            }

            if (opt.run_sync && opt.method_par_sync) {
                ParIluWorkspace w;
                const double ts = wall_seconds();
                if (parilu_setup(&sym, &w) != 0) {
                    ilu_symbolic_free(&sym);
                    csc_free(Apre);
                    csc_free(A);
                    return EXIT_FAILURE;
                }
                const double setup_time = wall_seconds() - ts;
                const double tf = wall_seconds();
                parilu_factor_sync(&w, opt.sweeps);
                const double factor_time = wall_seconds() - tf;
                print_row(name, "parilu", "sync", rep, nrep,
                          opt.k, opt.sweeps, threads,
                          preprocess_name, &pt,
                          opt.symbolic_mode_name,
                          Ause, &sym, &st, setup_time, factor_time,
                          ilu_workspace_checksum(w.L, w.U));
                parilu_free(&w);
            }

            if (opt.run_async && opt.method_par_async) {
                ParIluWorkspace w;
                const double ts = wall_seconds();
                if (parilu_setup(&sym, &w) != 0) {
                    ilu_symbolic_free(&sym);
                    csc_free(Apre);
                    csc_free(A);
                    return EXIT_FAILURE;
                }
                const double setup_time = wall_seconds() - ts;
                const double tf = wall_seconds();
                parilu_factor_async(&w, async_sweeps);
                const double factor_time = wall_seconds() - tf;
                print_row(name, "parilu", "async", rep, nrep,
                          opt.k, async_sweeps, threads,
                          preprocess_name, &pt,
                          opt.symbolic_mode_name,
                          Ause, &sym, &st, setup_time, factor_time,
                          ilu_workspace_checksum(w.L, w.U));
                parilu_free(&w);
            }
            }
        }

        ilu_symbolic_free(&sym);
        csc_free(Apre);
    }

    csc_free(A);
    free(opt.thread_list);
    return EXIT_SUCCESS;
}
