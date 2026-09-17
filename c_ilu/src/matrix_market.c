#include "csc.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int row;
    int col;
    double val;
} Triplet;

typedef enum {
    MM_GENERAL = 0,
    MM_SYMMETRIC = 1,
    MM_SKEW_SYMMETRIC = 2,
} MatrixMarketSymmetry;

typedef struct {
    int is_pattern;
    MatrixMarketSymmetry symmetry;
} MatrixMarketHeader;

static int triplet_cmp(const void *a, const void *b)
{
    const Triplet *ta = (const Triplet *)a;
    const Triplet *tb = (const Triplet *)b;
    if (ta->col != tb->col) {
        return ta->col < tb->col ? -1 : 1;
    }
    if (ta->row != tb->row) {
        return ta->row < tb->row ? -1 : 1;
    }
    return 0;
}

static int next_data_line(FILE *f, char *buf, size_t n)
{
    while (fgets(buf, (int)n, f)) {
        char *p = buf;
        while (isspace((unsigned char)*p)) {
            ++p;
        }
        if (*p == '\0' || *p == '%') {
            continue;
        }
        return 1;
    }
    return 0;
}

static void lower_ascii(char *s)
{
    for (; *s; ++s) {
        *s = (char)tolower((unsigned char)*s);
    }
}

static int parse_banner(const char *line, MatrixMarketHeader *header)
{
    char banner[64];
    char object[64];
    char format[64];
    char field[64];
    char symmetry[64];

    if (sscanf(line, "%63s %63s %63s %63s %63s",
               banner, object, format, field, symmetry) != 5) {
        return -1;
    }

    lower_ascii(banner);
    lower_ascii(object);
    lower_ascii(format);
    lower_ascii(field);
    lower_ascii(symmetry);

    if (strcmp(banner, "%%matrixmarket") != 0 ||
        strcmp(object, "matrix") != 0 ||
        strcmp(format, "coordinate") != 0) {
        return -1;
    }

    if (strcmp(field, "real") == 0 || strcmp(field, "integer") == 0) {
        header->is_pattern = 0;
    } else if (strcmp(field, "pattern") == 0) {
        header->is_pattern = 1;
    } else {
        return -2;
    }

    if (strcmp(symmetry, "general") == 0) {
        header->symmetry = MM_GENERAL;
    } else if (strcmp(symmetry, "symmetric") == 0) {
        header->symmetry = MM_SYMMETRIC;
    } else if (strcmp(symmetry, "skew-symmetric") == 0) {
        header->symmetry = MM_SKEW_SYMMETRIC;
    } else {
        return -3;
    }

    if (header->is_pattern && header->symmetry == MM_SKEW_SYMMETRIC) {
        return -4;
    }

    return 0;
}

CscMatrix *csc_read_matrix_market(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        return NULL;
    }

    char line[1024];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return NULL;
    }

    MatrixMarketHeader header = {0};
    const int banner_status = parse_banner(line, &header);
    if (banner_status == -1) {
        fprintf(stderr, "%s: only MatrixMarket matrix coordinate format is supported\n", path);
        fclose(f);
        return NULL;
    }
    if (banner_status == -2) {
        fprintf(stderr, "%s: only real, integer, and pattern MatrixMarket fields are supported\n", path);
        fclose(f);
        return NULL;
    }
    if (banner_status == -3) {
        fprintf(stderr, "%s: only general, symmetric, and skew-symmetric MatrixMarket symmetry is supported\n", path);
        fclose(f);
        return NULL;
    }
    if (banner_status == -4) {
        fprintf(stderr, "%s: pattern skew-symmetric MatrixMarket input is ambiguous and unsupported\n", path);
        fclose(f);
        return NULL;
    }

    if (!next_data_line(f, line, sizeof(line))) {
        fclose(f);
        return NULL;
    }

    int nrows = 0;
    int ncols = 0;
    int nnz_file = 0;
    if (sscanf(line, "%d %d %d", &nrows, &ncols, &nnz_file) != 3) {
        fprintf(stderr, "%s: malformed MatrixMarket dimensions\n", path);
        fclose(f);
        return NULL;
    }

    const int expands_symmetry = header.symmetry != MM_GENERAL;
    size_t cap = (size_t)nnz_file * (expands_symmetry ? 2u : 1u) + 1u;
    Triplet *trips = (Triplet *)ilu_xmalloc(cap * sizeof(Triplet));
    size_t nt = 0;

    for (int k = 0; k < nnz_file; ++k) {
        if (!next_data_line(f, line, sizeof(line))) {
            fprintf(stderr, "%s: unexpected end of file in entries\n", path);
            free(trips);
            fclose(f);
            return NULL;
        }

        int row = 0;
        int col = 0;
        double val = 1.0;
        const int nread = header.is_pattern ? sscanf(line, "%d %d", &row, &col)
                                            : sscanf(line, "%d %d %lf", &row, &col, &val);
        if ((!header.is_pattern && nread != 3) || (header.is_pattern && nread != 2)) {
            fprintf(stderr, "%s: malformed MatrixMarket entry\n", path);
            free(trips);
            fclose(f);
            return NULL;
        }

        row -= 1;
        col -= 1;
        if (row < 0 || row >= nrows || col < 0 || col >= ncols) {
            fprintf(stderr, "%s: MatrixMarket index out of bounds\n", path);
            free(trips);
            fclose(f);
            return NULL;
        }

        if (nt + 2 > cap) {
            cap *= 2;
            trips = (Triplet *)ilu_xrealloc(trips, cap * sizeof(Triplet));
        }
        trips[nt++] = (Triplet){.row = row, .col = col, .val = val};
        if (expands_symmetry && row != col) {
            const double reflected = header.symmetry == MM_SKEW_SYMMETRIC ? -val : val;
            trips[nt++] = (Triplet){.row = col, .col = row, .val = reflected};
        }
    }

    fclose(f);

    qsort(trips, nt, sizeof(Triplet), triplet_cmp);

    size_t nu = 0;
    for (size_t p = 0; p < nt;) {
        const int row = trips[p].row;
        const int col = trips[p].col;
        double val = 0.0;
        while (p < nt && trips[p].row == row && trips[p].col == col) {
            val += trips[p].val;
            p += 1;
        }
        trips[nu++] = (Triplet){.row = row, .col = col, .val = val};
    }

    CscMatrix *A = csc_alloc(nrows, ncols, (int)nu);
    for (size_t p = 0; p < nu; ++p) {
        A->colptr[trips[p].col + 1] += 1;
    }
    for (int j = 0; j < ncols; ++j) {
        A->colptr[j + 1] += A->colptr[j];
    }

    int *next = (int *)ilu_xmalloc((size_t)ncols * sizeof(int));
    memcpy(next, A->colptr, (size_t)ncols * sizeof(int));
    for (size_t p = 0; p < nu; ++p) {
        const int q = next[trips[p].col]++;
        A->rowind[q] = trips[p].row;
        A->x[q] = trips[p].val;
    }

    free(next);
    free(trips);

    if (!csc_validate_sorted(A)) {
        fprintf(stderr, "%s: internal CSC validation failed\n", path);
        csc_free(A);
        return NULL;
    }
    return A;
}
