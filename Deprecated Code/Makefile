CC=gcc
CFLAGS=-g -Wall --std=c99 -Wno-unknown-pragmas -O3
NFLAGS=-ccbin $(CC) -g -O3 -arch=compute_86 -code=sm_86
LIB=-lm
TARGETS=MM16 MM32 reduction

all: $(TARGETS)

MM16: tensor_core_matrix_multiply16.cu
	nvcc $(NFLAGS) -o $@ $< $(LIB)

MM32: tensor_core_matrix_multiply32.cu
	nvcc $(NFLAGS) -o $@ $< $(LIB)

reduction: reduct_slow_testing.cu
	nvcc $(NFLAGS) -o $@ $< $(LIB)

clean:
	rm -f $(TARGETS)