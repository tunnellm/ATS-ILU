#ifndef C_ILU_PREPROCESS_H
#define C_ILU_PREPROCESS_H

#include "csc.h"

typedef enum {
    PREPROCESS_NONE = 0,
    PREPROCESS_RCM = 1,
    PREPROCESS_MC64 = 2,
    PREPROCESS_MC64_RCM = 3,
    PREPROCESS_DIAG = 4,
    PREPROCESS_DIAG_RCM = 5,
    PREPROCESS_LEFT_DIAG = 6,
    PREPROCESS_LEFT_DIAG_RCM = 7,
} PreprocessMode;

typedef struct {
    PreprocessMode requested_mode;
    PreprocessMode effective_mode;
    double mc64;
    double mc64_prepare;
    double mc64_match;
    double mc64_apply;
    double diag_scale;
    double rcm;
    double rcm_order;
    double rcm_apply;
    double total;
    int diag_nonzeros_after_mc64;
    int rcm_levels_hint;
} PreprocessTimings;

typedef struct {
    int n;
    int *row_old_to_new;
    int *col_old_to_new;
    double *row_scale;
    double *col_scale;
} PreprocessTransform;

const char *preprocess_mode_name(PreprocessMode mode);
PreprocessMode preprocess_effective_mode(PreprocessMode mode);
void preprocess_warn_unavailable(void);
int parse_preprocess_mode(const char *name, PreprocessMode *mode);
CscMatrix *preprocess_matrix(const CscMatrix *A, PreprocessMode mode, PreprocessTimings *timings);
CscMatrix *preprocess_matrix_with_transform(
    const CscMatrix *A,
    PreprocessMode mode,
    PreprocessTimings *timings,
    PreprocessTransform *transform);
void preprocess_transform_free(PreprocessTransform *transform);

#endif
