CC ?= gcc
CFLAGS ?= -O3 -march=native -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude
LDFLAGS ?=
LDLIBS ?= -lm

OPENMP_CFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?= -fopenmp

SRC := src/csc.c src/matrix_market.c src/symbolic.c src/kernels.c src/preprocess.c

MC64_SOURCE ?=
MC64_LIBS ?=

ifneq ($(strip $(MC64_SOURCE)$(MC64_LIBS)),)
CPPFLAGS += -DC_ILU_HAVE_MC64
CFLAGS += -Wno-maybe-uninitialized -Wno-unused-parameter
LDLIBS += $(MC64_LIBS)
endif
ifneq ($(strip $(MC64_SOURCE)),)
ifeq ($(wildcard $(MC64_SOURCE)),)
$(error MC64_SOURCE does not exist: $(MC64_SOURCE))
endif
MC64_OBJ := mc64_external.o
endif

OBJ := $(SRC:.c=.o) $(MC64_OBJ)

mc64_external.o: $(MC64_SOURCE)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

.DEFAULT_GOAL := all

.PHONY: all check clean

all: bench_ilu dump_ilu_factors gmres_from_factors cg_from_factors

check: test_symbolic
	./test_symbolic

test_symbolic: tests/test_symbolic.o $(OBJ)
	$(CC) $(LDFLAGS) $(OPENMP_LDFLAGS) -o $@ $^ $(LDLIBS)

bench_ilu: tools/bench_ilu.o $(OBJ)
	$(CC) $(LDFLAGS) $(OPENMP_LDFLAGS) -o $@ $^ $(LDLIBS)

dump_ilu_factors: tools/dump_ilu_factors.o $(OBJ)
	$(CC) $(LDFLAGS) $(OPENMP_LDFLAGS) -o $@ $^ $(LDLIBS)

gmres_from_factors: tools/gmres_from_factors.o $(OBJ)
	$(CC) $(LDFLAGS) $(OPENMP_LDFLAGS) -o $@ $^ $(LDLIBS)

cg_from_factors: tools/cg_from_factors.o $(OBJ)
	$(CC) $(LDFLAGS) $(OPENMP_LDFLAGS) -o $@ $^ $(LDLIBS)

tools/cg_from_factors.o: tools/gmres_from_factors.c include/csc.h include/ilu.h include/preprocess.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP_CFLAGS) -DDEFAULT_SOLVER_CG -c -o $@ $<

%.o: %.c include/csc.h include/ilu.h include/preprocess.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OPENMP_CFLAGS) -c -o $@ $<

clean:
	rm -f bench_ilu dump_ilu_factors gmres_from_factors cg_from_factors test_symbolic tools/*.o tests/*.o src/*.o mc64_external.o
