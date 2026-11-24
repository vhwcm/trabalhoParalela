CC = mpicc
CFLAGS = -O3 -fopenmp -Wall

.PHONY: all debug seq_original limpar

all: lcs_paralelo

lcs_paralelo: lcs_paralelo.c
	$(CC) $(CFLAGS) -o lcs_paralelo lcs_paralelo.c

debug: lcs_paralelo_debug

lcs_paralelo_debug: lcs_paralelo.c
	$(CC) $(CFLAGS) -DDEBUGMATRIX -o lcs_paralelo_debug lcs_paralelo.c

seq_original: lcs_ref lcs_ref_debug

lcs_ref: ../lcs_source_code/lcs.c
	gcc -O3 -o lcs_ref ../lcs_source_code/lcs.c

lcs_ref_debug: ../lcs_source_code/lcs.c
	gcc -O3 -DDEBUGMATRIX -o lcs_ref_debug ../lcs_source_code/lcs.c

limpar:
	rm -f lcs_paralelo lcs_paralelo_debug lcs_ref lcs_ref_debug
