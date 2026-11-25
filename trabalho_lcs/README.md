# Trabalho 1 - LCS com OpenMP e MPI

CI1316 - Programação Paralela - UFPR

Felipe Augusto Dittert Noleto (GRR20205689)
Victor Hugo Weigmann Chequer Maia (GRR20245275)

## Arquivos

- `lcs_paralelo.c`: todas as versões (seq, omp_diag, omp_blocos, omp_tarefas, mpi_blocos e hibrido), escolhidas pela flag `-v`.
- `Makefile`: compila o programa paralelo, a versão de depuração e o `lcs.c` original.

## Compilar

Precisa do `mpicc` (Open MPI) com suporte a OpenMP.

```
make                # gera ./lcs_paralelo
make debug          # gera ./lcs_paralelo_debug, que imprime a matriz (DEBUGMATRIX)
make seq_original   # compila o lcs.c original (espera ele em ../lcs_source_code)
make limpar
```

## Executar

Por padrão o programa lê `fileA.in` e `fileB.in` da pasta atual e imprime o Score, igual ao código original.

```
./lcs_paralelo                                # omp_tarefas com todas as threads
OMP_NUM_THREADS=4 ./lcs_paralelo -v omp_blocos
mpirun -np 4 ./lcs_paralelo                   # com mais de 1 processo usa o hibrido
mpirun -np 4 ./lcs_paralelo -v mpi_blocos -a A.in -B B.in
```

Opções:

- `-v versao`: `seq`, `omp_diag`, `omp_blocos`, `omp_tarefas`, `mpi_blocos` ou `hibrido`.
- `-b tamanho`: tamanho do bloco (padrão 1024).
- `-a arquivo` e `-B arquivo`: arquivos das sequências A e B.
- `-m`: imprime no stderr o tempo de cada fase (leitura, alocação, inicialização, cálculo e liberação).
