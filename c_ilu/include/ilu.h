#ifndef C_ILU_ILU_H
#define C_ILU_ILU_H

#include "csc.h"

typedef struct {
    double symbolic_pattern;
    double numeric_fill;
    double transform;
    double total;
    int symbolic_levels;
    int symbolic_fallback;
} IluSymbolicTimings;

typedef enum {
    ILU_SYMBOLIC_SERIAL = 0,
    ILU_SYMBOLIC_LEVELSET = 1,
} IluSymbolicMode;

typedef enum {
    ATS_SCALE_MATERIALIZE = 0,
    ATS_SCALE_FOLDED = 1,
} AtsScaleMode;

typedef struct {
    CscMatrix *Lsym;   /* Transposed lower factor pattern and values. */
    CscMatrix *Usym;   /* Upper factor pattern and values. */
    CscMatrix *S;      /* Union pattern of untransposed L and U, with A values copied in. */
    CscMatrix *Lsymm1; /* Strict upper part of Lsym, used by ParILU as unit lower storage. */
} IluSymbolic;

typedef struct {
    CscMatrix *L;
    CscMatrix *U;
} SeqIluWorkspace;

typedef struct {
    const CscMatrix *Lsrc_mat;
    const CscMatrix *Usrc_mat;
    const double *Lsrc;
    const double *Usrc;
    CscMatrix *L;
    CscMatrix *U;
    double *Ltmp;
    double *Utmp;
    double *D;
} AtsIluWorkspace;

typedef struct {
    const CscMatrix *Lsrc_mat;
    const double *Lsrc;
    CscMatrix *L;
    double *Ltmp;
    double diagonal_shift;
} AtsIcWorkspace;

typedef struct {
    const CscMatrix *Lsrc_mat;
    const double *Lsrc;
    CscMatrix *L;
    int *entry_row;
} ParIcWorkspace;

typedef struct {
    const CscMatrix *S;
    const int *I;
    int *J;
    const double *V;
    int *out_pos;
    int *diag_pos;
    CscMatrix *L;
    CscMatrix *U;
    double *Ltmp;
    double *Utmp;
} ParIluWorkspace;

int ilu_build_symbolic(const CscMatrix *A, int k, IluSymbolic *sym, IluSymbolicTimings *timings);
int ilu_build_symbolic_mode(
    const CscMatrix *A,
    int k,
    IluSymbolicMode mode,
    IluSymbolic *sym,
    IluSymbolicTimings *timings);
void ilu_symbolic_free(IluSymbolic *sym);

int seq_ilu_setup(const IluSymbolic *sym, SeqIluWorkspace *w);
void seq_ilu_free(SeqIluWorkspace *w);
void seq_ilu_factor(SeqIluWorkspace *w);

int ats_ilu_setup(const IluSymbolic *sym, AtsIluWorkspace *w);
int ats_ilu_setup_async(const IluSymbolic *sym, AtsIluWorkspace *w);
void ats_ilu_free(AtsIluWorkspace *w);
void ats_ilu_factor_sync_mode(AtsIluWorkspace *w, int sweeps, AtsScaleMode mode);
void ats_ilu_factor_sync(AtsIluWorkspace *w, int sweeps);
void ats_ilu_factor_sync_sweep_mode(AtsIluWorkspace *w, AtsScaleMode mode);
void ats_ilu_factor_async_begin(AtsIluWorkspace *w);
void ats_ilu_factor_async_sweep(AtsIluWorkspace *w);
void ats_ilu_factor_async(AtsIluWorkspace *w, int sweeps);

int ats_ic_setup_async(
    const IluSymbolic *sym,
    AtsIcWorkspace *w,
    double diagonal_shift);
int ats_ic_setup_sequential(
    const IluSymbolic *sym,
    AtsIcWorkspace *w,
    double diagonal_shift);
void ats_ic_free(AtsIcWorkspace *w);
int ats_ic_factor_sequential(AtsIcWorkspace *w);
int ats_ic_factor_async_sweep(AtsIcWorkspace *w);
int ats_ic_factor_async(AtsIcWorkspace *w, int sweeps);

int paric_setup(const IluSymbolic *sym, ParIcWorkspace *w);
void paric_free(ParIcWorkspace *w);
int paric_factor_async_sweep(ParIcWorkspace *w);
int paric_factor_async(ParIcWorkspace *w, int sweeps);

int parilu_setup(const IluSymbolic *sym, ParIluWorkspace *w);
void parilu_free(ParIluWorkspace *w);
void parilu_factor_sync_sweep(ParIluWorkspace *w);
void parilu_factor_async_sweep(ParIluWorkspace *w);
void parilu_factor_sync(ParIluWorkspace *w, int sweeps);
void parilu_factor_async(ParIluWorkspace *w, int sweeps);

double ilu_workspace_checksum(const CscMatrix *L, const CscMatrix *U);

#endif
