#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

typedef unsigned short tipo_matriz;

char *ler_sequencia(const char *nome_arquivo, int *tamanho) {
    FILE *arquivo = fopen(nome_arquivo, "rb");
    if (arquivo == NULL) {
        fprintf(stderr, "Erro ao abrir %s\n", nome_arquivo);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fseek(arquivo, 0, SEEK_END);
    long tam_arquivo = ftell(arquivo);
    rewind(arquivo);
    char *texto = malloc((size_t)(tam_arquivo + 1) * sizeof(char));
    if (texto == NULL) {
        fprintf(stderr, "Erro de alocacao\n");
        fclose(arquivo);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t lidos = fread(texto, 1, (size_t)tam_arquivo, arquivo);
    fclose(arquivo);
    texto[lidos] = '\0';
    while (lidos > 0 && (texto[lidos - 1] == '\n' || texto[lidos - 1] == '\r')) {
        lidos--;
        texto[lidos] = '\0';
    }
    *tamanho = (int)lidos;
    return texto;
}

void imprimir_matriz(char *seq_a, char *seq_b, tipo_matriz *matriz, int linhas, int colunas, int linha_inicio) {
    printf("Matriz de Pontuacao:\n");
    printf("========================================\n");
    printf("    ");
    printf("%5c   ", ' ');
    for (int j = 0; j < colunas; j++) {
        printf("%5c   ", seq_b[j]);
    }
    printf("\n");
    for (int i = 0; i <= linhas; i++) {
        if (i == 0) {
            printf("    ");
        } else {
            printf("%c   ", seq_a[linha_inicio + i - 2]);
        }
        for (int j = 0; j <= colunas; j++) {
            printf("%5d   ", matriz[(size_t)i * (colunas + 1) + j]);
        }
        printf("\n");
    }
    printf("========================================\n");
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int meu_ranque = 0;
    int num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_ranque);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int tam_a = 0;
    int tam_b = 0;
    char *seq_a = NULL;
    char *seq_b = NULL;

    if (meu_ranque == 0) {
        seq_a = ler_sequencia("fileA.in", &tam_a);
        seq_b = ler_sequencia("fileB.in", &tam_b);
    }

    MPI_Bcast(&tam_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tam_b, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (meu_ranque != 0) {
        seq_a = malloc((size_t)(tam_a + 1) * sizeof(char));
        seq_b = malloc((size_t)(tam_b + 1) * sizeof(char));
        if (seq_a == NULL || seq_b == NULL) {
            fprintf(stderr, "Erro de alocacao\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(seq_a, tam_a + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(seq_b, tam_b + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (tam_a == 0 || tam_b == 0) {
        if (meu_ranque == 0) {
            printf("\nScore: 0\n");
        }
        free(seq_a);
        free(seq_b);
        MPI_Finalize();
        return 0;
    }

    int procs_ativos = (num_procs < tam_a) ? num_procs : tam_a;
    if (meu_ranque >= procs_ativos) {
        free(seq_a);
        free(seq_b);
        MPI_Finalize();
        return 0;
    }

    int base = tam_a / procs_ativos;
    int resto = tam_a % procs_ativos;
    int linhas_locais = base + (meu_ranque < resto ? 1 : 0);
    int linha_inicio = meu_ranque * base + (meu_ranque < resto ? meu_ranque : resto) + 1;

    tipo_matriz *matriz = calloc((size_t)(linhas_locais + 1) * (tam_b + 1), sizeof(tipo_matriz));
    if (matriz == NULL) {
        fprintf(stderr, "Erro ao alocar matriz\n");
        free(seq_a);
        free(seq_b);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int tam_bloco = 64;
    for (int col_inicio = 1; col_inicio <= tam_b; col_inicio += tam_bloco) {
        int col_fim = col_inicio + tam_bloco - 1;
        if (col_fim > tam_b) {
            col_fim = tam_b;
        }
        int larg_bloco = col_fim - col_inicio + 1;

        if (meu_ranque > 0) {
            MPI_Recv(&matriz[col_inicio], larg_bloco, MPI_UNSIGNED_SHORT, meu_ranque - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        #pragma omp parallel
        {
            for (int d = 0; d <= linhas_locais + larg_bloco - 2; d++) {
                int li_min = (d - larg_bloco + 1 > 0) ? (d - larg_bloco + 1) : 0;
                int li_max = (d < linhas_locais - 1) ? d : (linhas_locais - 1);
                #pragma omp for
                for (int li = li_min; li <= li_max; li++) {
                    int lj = d - li;
                    int i = li + 1;
                    int j = col_inicio + lj;
                    if (seq_a[linha_inicio + i - 2] == seq_b[j - 1]) {
                        matriz[(size_t)i * (tam_b + 1) + j] = matriz[(size_t)(i - 1) * (tam_b + 1) + (j - 1)] + 1;
                    } else {
                        tipo_matriz cima = matriz[(size_t)(i - 1) * (tam_b + 1) + j];
                        tipo_matriz esquerda = matriz[(size_t)i * (tam_b + 1) + (j - 1)];
                        matriz[(size_t)i * (tam_b + 1) + j] = (cima > esquerda) ? cima : esquerda;
                    }
                }
            }
        }

        if (meu_ranque < procs_ativos - 1) {
            MPI_Send(&matriz[(size_t)linhas_locais * (tam_b + 1) + col_inicio], larg_bloco, MPI_UNSIGNED_SHORT, meu_ranque + 1, 0, MPI_COMM_WORLD);
        }
    }

#ifdef DEPURAR_MATRIZ
    imprimir_matriz(seq_a, seq_b, matriz, linhas_locais, tam_b, linha_inicio);
#endif

    tipo_matriz pontuacao = matriz[(size_t)linhas_locais * (tam_b + 1) + tam_b];
    if (procs_ativos == 1) {
        printf("\nScore: %d\n", pontuacao);
    } else if (meu_ranque == procs_ativos - 1) {
        MPI_Send(&pontuacao, 1, MPI_UNSIGNED_SHORT, 0, 1, MPI_COMM_WORLD);
    } else if (meu_ranque == 0) {
        MPI_Recv(&pontuacao, 1, MPI_UNSIGNED_SHORT, procs_ativos - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("\nScore: %d\n", pontuacao);
    }

    free(matriz);
    free(seq_a);
    free(seq_b);
    MPI_Finalize();
    return 0;
}
