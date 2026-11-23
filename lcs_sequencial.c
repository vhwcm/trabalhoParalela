#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned short tipo_matriz;

char *ler_sequencia(const char *nome_arquivo, int *tamanho) {
    FILE *arquivo = fopen(nome_arquivo, "rb");
    if (arquivo == NULL) {
        fprintf(stderr, "Erro ao abrir %s\n", nome_arquivo);
        exit(1);
    }
    fseek(arquivo, 0, SEEK_END);
    long tam_arquivo = ftell(arquivo);
    rewind(arquivo);
    char *texto = malloc((size_t)(tam_arquivo + 1) * sizeof(char));
    if (texto == NULL) {
        fprintf(stderr, "Erro de alocacao\n");
        fclose(arquivo);
        exit(1);
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

void imprimir_matriz(char *seq_a, char *seq_b, tipo_matriz *matriz, int tam_a, int tam_b) {
    printf("Matriz de Pontuacao:\n");
    printf("========================================\n");
    printf("    ");
    printf("%5c   ", ' ');
    for (int j = 0; j < tam_b; j++) {
        printf("%5c   ", seq_b[j]);
    }
    printf("\n");
    for (int i = 0; i <= tam_a; i++) {
        if (i == 0) {
            printf("    ");
        } else {
            printf("%c   ", seq_a[i - 1]);
        }
        for (int j = 0; j <= tam_b; j++) {
            printf("%5d   ", matriz[(size_t)i * (tam_b + 1) + j]);
        }
        printf("\n");
    }
    printf("========================================\n");
}

double calcular_diferenca_tempo(struct timespec inicio, struct timespec fim) {
    return (double)(fim.tv_sec - inicio.tv_sec) + (double)(fim.tv_nsec - inicio.tv_nsec) / 1e9;
}

int main(int argc, char *argv[]) {
    int medir_tempo = 0;
    for (int k = 1; k < argc; k++) {
        if (strcmp(argv[k], "-t") == 0) {
            medir_tempo = 1;
        }
    }

    struct timespec inicio_total, fim_total, inicio_kernel, fim_kernel;
    if (medir_tempo) {
        clock_gettime(CLOCK_MONOTONIC, &inicio_total);
    }

    int tam_a = 0;
    int tam_b = 0;
    char *seq_a = ler_sequencia("fileA.in", &tam_a);
    char *seq_b = ler_sequencia("fileB.in", &tam_b);

    tipo_matriz *matriz = calloc((size_t)(tam_a + 1) * (tam_b + 1), sizeof(tipo_matriz));
    if (matriz == NULL) {
        fprintf(stderr, "Erro ao alocar matriz\n");
        free(seq_a);
        free(seq_b);
        return 1;
    }

    if (medir_tempo) {
        clock_gettime(CLOCK_MONOTONIC, &inicio_kernel);
    }

    for (int i = 1; i <= tam_a; i++) {
        for (int j = 1; j <= tam_b; j++) {
            if (seq_a[i - 1] == seq_b[j - 1]) {
                matriz[(size_t)i * (tam_b + 1) + j] = matriz[(size_t)(i - 1) * (tam_b + 1) + (j - 1)] + 1;
            } else {
                tipo_matriz cima = matriz[(size_t)(i - 1) * (tam_b + 1) + j];
                tipo_matriz esquerda = matriz[(size_t)i * (tam_b + 1) + (j - 1)];
                matriz[(size_t)i * (tam_b + 1) + j] = (cima > esquerda) ? cima : esquerda;
            }
        }
    }

    if (medir_tempo) {
        clock_gettime(CLOCK_MONOTONIC, &fim_kernel);
    }

#ifdef DEPURAR_MATRIZ
    imprimir_matriz(seq_a, seq_b, matriz, tam_a, tam_b);
#endif

    tipo_matriz pontuacao = matriz[(size_t)tam_a * (tam_b + 1) + tam_b];
    printf("\nScore: %d\n", pontuacao);

    free(matriz);
    free(seq_a);
    free(seq_b);

    if (medir_tempo) {
        clock_gettime(CLOCK_MONOTONIC, &fim_total);
        double tempo_total = calcular_diferenca_tempo(inicio_total, fim_total);
        double tempo_kernel = calcular_diferenca_tempo(inicio_kernel, fim_kernel);
        double tempo_sequencial = tempo_total - tempo_kernel;
        fprintf(stderr, "Tempo total: %.6f s\n", tempo_total);
        fprintf(stderr, "Tempo sequencial: %.6f s\n", tempo_sequencial);
        fprintf(stderr, "Tempo kernel: %.6f s\n", tempo_kernel);
    }

    return 0;
}
