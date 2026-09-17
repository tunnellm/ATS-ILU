#ifndef ATS_MC64_COMPAT_H
#define ATS_MC64_COMPAT_H
/* Interface shim for a separately supplied 32-bit mc64ad_dist.c translation.
 * This file contains no MC64 implementation. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
typedef int int_t;
static inline double dmach_dist(char *c)
{
    switch (c ? *c : '\0') {
    case 'E': case 'e': return DBL_EPSILON;
    case 'P': case 'p': return DBL_EPSILON * FLT_RADIX;
    case 'S': case 's': case 'U': case 'u': return DBL_MIN;
    case 'B': case 'b': return FLT_RADIX;
    case 'N': case 'n': return DBL_MANT_DIG;
    case 'R': case 'r': return 1;
    case 'M': case 'm': return DBL_MIN_EXP;
    case 'L': case 'l': return DBL_MAX_EXP;
    default: return DBL_MAX;
    }
}
#endif
