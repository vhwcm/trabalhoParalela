CC = gcc
MPICC = mpicc
CFLAGS = -O3 -Wall
MPIFLAGS = -O3 -fopenmp -Wall

all: lcs lcs_sequencial lcs_paralelo

lcs: lcs_source_code/lcs.c
	$(CC) $(CFLAGS) -o lcs lcs_source_code/lcs.c

lcs_sequencial: lcs_sequencial.c
	$(CC) $(CFLAGS) -o lcs_sequencial lcs_sequencial.c

lcs_paralelo: lcs_paralelo.c
	$(MPICC) $(MPIFLAGS) -o lcs_paralelo lcs_paralelo.c

clean:
	rm -f lcs lcs_sequencial lcs_paralelo
