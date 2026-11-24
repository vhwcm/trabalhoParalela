#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

typedef unsigned short tipo_matriz;

char *ler_sequencia(const char *nome_arquivo) {
	FILE *arq = fopen(nome_arquivo, "rt");
	if (arq == NULL) {
		fprintf(stderr, "Erro ao ler arquivo %s\n", nome_arquivo);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	fseek(arq, 0L, SEEK_END);
	long tamanho = ftell(arq);
	rewind(arq);

	char *seq = (char *)calloc((size_t)tamanho + 1, sizeof(char));
	if (seq == NULL) {
		fprintf(stderr, "Erro ao alocar memoria para sequencia %s.\n", nome_arquivo);
		fclose(arq);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	size_t i = 0;
	int c;
	while ((c = fgetc(arq)) != EOF) {
		if (c != '\n') {
			seq[i++] = (char)c;
		}
	}
	seq[i] = '\0';

	fclose(arq);
	return seq;
}

tipo_matriz **alocar_matriz(int tamA, int tamB) {
	size_t linhas = (size_t)tamB + 1;
	size_t colunas = (size_t)tamA + 1;

	tipo_matriz **matriz = (tipo_matriz **)malloc(linhas * sizeof(tipo_matriz *));
	// Bloco unico contiguo usando size_t para evitar overflow quando N >= 46341
	tipo_matriz *bloco = (tipo_matriz *)malloc(linhas * colunas * sizeof(tipo_matriz));
	if (matriz == NULL || bloco == NULL) {
		fprintf(stderr, "Erro ao alocar matriz de pontuacao.\n");
		free(matriz);
		free(bloco);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	for (size_t i = 0; i < linhas; i++) {
		matriz[i] = bloco + i * colunas;
	}
	return matriz;
}

void inicializar_matriz(tipo_matriz **matriz, int tamA, int tamB) {
	for (int j = 0; j < tamA + 1; j++) {
		matriz[0][j] = 0;
	}
	for (int i = 1; i < tamB + 1; i++) {
		matriz[i][0] = 0;
	}
}

void liberar_matriz(tipo_matriz **matriz) {
	if (matriz != NULL) {
		// matriz[0] aponta para o inicio do bloco contiguo alocado em alocar_matriz
		free(matriz[0]);
		free(matriz);
	}
}

void imprimir_matriz(char *seqA, char *seqB, tipo_matriz **matriz, int tamA, int tamB) {
	printf("Score Matrix:\n");
	printf("========================================\n");

	printf("    ");
	printf("%5c   ", ' ');

	for (int j = 0; j < tamA; j++) {
		printf("%5c   ", seqA[j]);
	}
	printf("\n");
	for (int i = 0; i < tamB + 1; i++) {
		if (i == 0) {
			printf("    ");
		} else {
			printf("%c   ", seqB[i - 1]);
		}
		for (int j = 0; j < tamA + 1; j++) {
			printf("%5d   ", matriz[i][j]);
		}
		printf("\n");
	}
	printf("========================================\n");
}

tipo_matriz lcs_sequencial(tipo_matriz **matriz, int tamA, int tamB, char *seqA, char *seqB) {
	if (tamA == 0 || tamB == 0) {
		return 0;
	}

	for (int i = 1; i < tamB + 1; i++) {
		for (int j = 1; j < tamA + 1; j++) {
			if (seqA[j - 1] == seqB[i - 1]) {
				matriz[i][j] = matriz[i - 1][j - 1] + 1;
			} else {
				matriz[i][j] = max(matriz[i - 1][j], matriz[i][j - 1]);
			}
		}
	}
	return matriz[tamB][tamA];
}

int lcs_omp_diagonal(tipo_matriz **matriz, int tamA, int tamB, char *seqA, char *seqB) {
	if (tamA == 0 || tamB == 0) {
		return 0;
	}

	// Percorre antidiagonais i + j = d; celulas na mesma diagonal sao independentes
	for (int d = 2; d <= tamA + tamB; d++) {
		int i_ini = (d - tamA > 1) ? (d - tamA) : 1;
		int i_fim = (d - 1 < tamB) ? (d - 1) : tamB;

		#pragma omp parallel for schedule(static)
		for (int i = i_ini; i <= i_fim; i++) {
			int j = d - i;
			if (seqA[j - 1] == seqB[i - 1]) {
				matriz[i][j] = matriz[i - 1][j - 1] + 1;
			} else {
				matriz[i][j] = max(matriz[i - 1][j], matriz[i][j - 1]);
			}
		}
	}
	return matriz[tamB][tamA];
}

static inline void processar_bloco(tipo_matriz **matriz, char *seqA, char *seqB,
		int i_ini, int i_fim, int j_ini, int j_fim) {
	for (int i = i_ini; i <= i_fim; i++) {
		char cB = seqB[i - 1];
		for (int j = j_ini; j <= j_fim; j++) {
			if (seqA[j - 1] == cB) {
				matriz[i][j] = matriz[i - 1][j - 1] + 1;
			} else {
				matriz[i][j] = max(matriz[i - 1][j], matriz[i][j - 1]);
			}
		}
	}
}

int lcs_omp_blocos(tipo_matriz **matriz, int tamA, int tamB, char *seqA, char *seqB, int tam_bloco) {
	if (tamA == 0 || tamB == 0) {
		return 0;
	}

	int num_bi = (tamB + tam_bloco - 1) / tam_bloco;
	int num_bj = (tamA + tam_bloco - 1) / tam_bloco;

	// Percorre antidiagonais de blocos; blocos parciais nas bordas justificam schedule(dynamic, 1)
	for (int db = 0; db <= num_bi + num_bj - 2; db++) {
		int bi_ini = (db - num_bj + 1 > 0) ? (db - num_bj + 1) : 0;
		int bi_fim = (db < num_bi - 1) ? db : (num_bi - 1);

		#pragma omp parallel for schedule(dynamic, 1)
		for (int bi = bi_ini; bi <= bi_fim; bi++) {
			int bj = db - bi;
			int i_ini = bi * tam_bloco + 1;
			int i_fim = (i_ini + tam_bloco - 1 < tamB) ? (i_ini + tam_bloco - 1) : tamB;
			int j_ini = bj * tam_bloco + 1;
			int j_fim = (j_ini + tam_bloco - 1 < tamA) ? (j_ini + tam_bloco - 1) : tamA;
			processar_bloco(matriz, seqA, seqB, i_ini, i_fim, j_ini, j_fim);
		}
	}
	return matriz[tamB][tamA];
}

int lcs_omp_tarefas(tipo_matriz **matriz, int tamA, int tamB, char *seqA, char *seqB, int tam_bloco) {
	if (tamA == 0 || tamB == 0) {
		return 0;
	}

	int num_bi = (tamB + tam_bloco - 1) / tam_bloco;
	int num_bj = (tamA + tam_bloco - 1) / tam_bloco;
	int colunas_dep = num_bj + 1;

	// Matriz sentinela 1-based para sinalizar dependencias entre blocos (acima e a esquerda)
	char *dep = (char *)calloc((size_t)(num_bi + 1) * (size_t)colunas_dep, sizeof(char));
	if (dep == NULL) {
		fprintf(stderr, "Erro ao alocar matriz de dependencias.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	#pragma omp parallel
	{
		#pragma omp single
		{
			for (int bi = 0; bi < num_bi; bi++) {
				for (int bj = 0; bj < num_bj; bj++) {
					#pragma omp task firstprivate(bi, bj) \
						depend(in: dep[(size_t)bi * colunas_dep + (bj + 1)], dep[(size_t)(bi + 1) * colunas_dep + bj]) \
						depend(out: dep[(size_t)(bi + 1) * colunas_dep + (bj + 1)])
					{
						int i_ini = bi * tam_bloco + 1;
						int i_fim = (i_ini + tam_bloco - 1 < tamB) ? (i_ini + tam_bloco - 1) : tamB;
						int j_ini = bj * tam_bloco + 1;
						int j_fim = (j_ini + tam_bloco - 1 < tamA) ? (j_ini + tam_bloco - 1) : tamA;
						processar_bloco(matriz, seqA, seqB, i_ini, i_fim, j_ini, j_fim);
					}
				}
			}
		}
	}

	free(dep);
	return matriz[tamB][tamA];
}

int lcs_mpi_blocos(tipo_matriz **matriz_local, int tamA, int tamB, char *seqA, char *seqB,
		int tam_bloco, int meu_ranque, int num_procs) {
	if (tamA == 0 || tamB == 0) {
		return 0;
	}

	// Se houver mais processos do que linhas, os excedentes ficam ociosos
	int procs_ativos = (tamB < num_procs) ? tamB : num_procs;
	if (meu_ranque >= procs_ativos) {
		return 0;
	}

	int base = tamB / procs_ativos;
	int resto = tamB % procs_ativos;
	int linhas_locais = base + (meu_ranque < resto ? 1 : 0);
	int linha_inicio = meu_ranque * base + (meu_ranque < resto ? meu_ranque : resto);

	// Pipeline de blocos de coluna: cada rank envia sua ultima linha como borda para o proximo
	for (int j_ini = 1, tag = 0; j_ini <= tamA; j_ini += tam_bloco, tag++) {
		int j_fim = (j_ini + tam_bloco - 1 < tamA) ? (j_ini + tam_bloco - 1) : tamA;
		int largura = j_fim - j_ini + 1;

		if (meu_ranque > 0) {
			MPI_Recv(&matriz_local[0][j_ini], largura, MPI_UNSIGNED_SHORT,
					meu_ranque - 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		for (int i = 1; i <= linhas_locais; i++) {
			char cB = seqB[linha_inicio + i - 1];
			for (int j = j_ini; j <= j_fim; j++) {
				if (seqA[j - 1] == cB) {
					matriz_local[i][j] = matriz_local[i - 1][j - 1] + 1;
				} else {
					matriz_local[i][j] = max(matriz_local[i - 1][j], matriz_local[i][j - 1]);
				}
			}
		}

		if (meu_ranque < procs_ativos - 1) {
			MPI_Send(&matriz_local[linhas_locais][j_ini], largura, MPI_UNSIGNED_SHORT,
					meu_ranque + 1, tag, MPI_COMM_WORLD);
		}
	}

	int pontuacao = 0;
	if (procs_ativos == 1) {
		pontuacao = (int)matriz_local[linhas_locais][tamA];
	} else if (meu_ranque == procs_ativos - 1) {
		pontuacao = (int)matriz_local[linhas_locais][tamA];
		MPI_Send(&pontuacao, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	} else if (meu_ranque == 0) {
		MPI_Recv(&pontuacao, 1, MPI_INT, procs_ativos - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	return pontuacao;
}

int lcs_hibrido(tipo_matriz **matriz_local, int tamA, int tamB, char *seqA, char *seqB,
		int tam_bloco, int meu_ranque, int num_procs) {
	if (tamA == 0 || tamB == 0) {
		return 0;
	}

	int procs_ativos = (tamB < num_procs) ? tamB : num_procs;
	if (meu_ranque >= procs_ativos) {
		return 0;
	}

	int base = tamB / procs_ativos;
	int resto = tamB % procs_ativos;
	int linhas_locais = base + (meu_ranque < resto ? 1 : 0);
	int linha_inicio = meu_ranque * base + (meu_ranque < resto ? meu_ranque : resto);

	int max_threads = omp_get_max_threads();
	// Com multiplas threads, blocos MPI mais largos permitem wavefront interno em OpenMP
	int largura_mpi = (max_threads > 1)
			? ((tam_bloco > tamA / 4) ? tamA : (tam_bloco * 4))
			: tam_bloco;
	if (largura_mpi < 1) {
		largura_mpi = 1;
	}

	for (int j_ini = 1, tag = 0; j_ini <= tamA; j_ini += largura_mpi, tag++) {
		int j_fim = (j_ini + largura_mpi - 1 < tamA) ? (j_ini + largura_mpi - 1) : tamA;
		int largura = j_fim - j_ini + 1;

		if (meu_ranque > 0) {
			MPI_Recv(&matriz_local[0][j_ini], largura, MPI_UNSIGNED_SHORT,
					meu_ranque - 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		if (max_threads <= 1 || linhas_locais <= tam_bloco || largura <= tam_bloco) {
			processar_bloco(matriz_local, seqA, seqB + linha_inicio, 1, linhas_locais, j_ini, j_fim);
		} else {
			// Divide o bloco MPI em sub-blocos tam_bloco x tam_bloco em antidiagonais OpenMP
			int num_bi = (linhas_locais + tam_bloco - 1) / tam_bloco;
			int num_bj = (largura + tam_bloco - 1) / tam_bloco;

			for (int db = 0; db <= num_bi + num_bj - 2; db++) {
				int bi_ini = (db - num_bj + 1 > 0) ? (db - num_bj + 1) : 0;
				int bi_fim = (db < num_bi - 1) ? db : (num_bi - 1);

				#pragma omp parallel for schedule(dynamic, 1)
				for (int bi = bi_ini; bi <= bi_fim; bi++) {
					int bj = db - bi;
					int i0 = bi * tam_bloco + 1;
					int i1 = (i0 + tam_bloco - 1 < linhas_locais) ? (i0 + tam_bloco - 1) : linhas_locais;
					int j0 = j_ini + bj * tam_bloco;
					int j1 = (j0 + tam_bloco - 1 < j_fim) ? (j0 + tam_bloco - 1) : j_fim;
					processar_bloco(matriz_local, seqA, seqB + linha_inicio, i0, i1, j0, j1);
				}
			}
		}

		if (meu_ranque < procs_ativos - 1) {
			MPI_Send(&matriz_local[linhas_locais][j_ini], largura, MPI_UNSIGNED_SHORT,
					meu_ranque + 1, tag, MPI_COMM_WORLD);
		}
	}

	int pontuacao = 0;
	if (procs_ativos == 1) {
		pontuacao = (int)matriz_local[linhas_locais][tamA];
	} else if (meu_ranque == procs_ativos - 1) {
		pontuacao = (int)matriz_local[linhas_locais][tamA];
		MPI_Send(&pontuacao, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	} else if (meu_ranque == 0) {
		MPI_Recv(&pontuacao, 1, MPI_INT, procs_ativos - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	return pontuacao;
}

int main(int argc, char **argv) {
	int suporte = 0;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &suporte);

	int meu_ranque = 0;
	int num_procs = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &meu_ranque);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	// Estrategia padrao quando -v nao e informada: hibrido com multiplos processos ou omp_tarefas
	const char *versao = (num_procs > 1) ? "hibrido" : "omp_tarefas";
	int tam_bloco = 1024;
	const char *arqA = "fileA.in";
	const char *arqB = "fileB.in";
	int medir_tempos = 0;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
			versao = argv[++i];
		} else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
			tam_bloco = atoi(argv[++i]);
		} else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
			arqA = argv[++i];
		} else if (strcmp(argv[i], "-B") == 0 && i + 1 < argc) {
			arqB = argv[++i];
		} else if (strcmp(argv[i], "-m") == 0) {
			medir_tempos = 1;
		} else {
			if (meu_ranque == 0) {
				fprintf(stderr, "Uso: %s [-v versao] [-b tam_bloco] [-a arqA] [-B arqB] [-m]\n", argv[0]);
			}
			MPI_Finalize();
			return EXIT_FAILURE;
		}
	}

	if (tam_bloco < 1) {
		tam_bloco = 1;
	}

	(void)suporte;

	if (strcmp(versao, "seq") == 0 || strcmp(versao, "omp_diag") == 0 ||
			strcmp(versao, "omp_blocos") == 0 || strcmp(versao, "omp_tarefas") == 0) {
		if (meu_ranque == 0) {
			double t_inicio_total = MPI_Wtime();
			double t0;

			t0 = MPI_Wtime();
			char *seqA = ler_sequencia(arqA);
			char *seqB = ler_sequencia(arqB);
			int tamA = (int)strlen(seqA);
			int tamB = (int)strlen(seqB);
			double t_leitura = MPI_Wtime() - t0;

			t0 = MPI_Wtime();
			tipo_matriz **matriz = alocar_matriz(tamA, tamB);
			double t_alocacao = MPI_Wtime() - t0;

			t0 = MPI_Wtime();
			inicializar_matriz(matriz, tamA, tamB);
			double t_inicializacao = MPI_Wtime() - t0;

			t0 = MPI_Wtime();
			int pontuacao = 0;
			if (strcmp(versao, "seq") == 0) {
				pontuacao = lcs_sequencial(matriz, tamA, tamB, seqA, seqB);
			} else if (strcmp(versao, "omp_diag") == 0) {
				pontuacao = lcs_omp_diagonal(matriz, tamA, tamB, seqA, seqB);
			} else if (strcmp(versao, "omp_blocos") == 0) {
				pontuacao = lcs_omp_blocos(matriz, tamA, tamB, seqA, seqB, tam_bloco);
			} else {
				pontuacao = lcs_omp_tarefas(matriz, tamA, tamB, seqA, seqB, tam_bloco);
			}
			double t_calculo = MPI_Wtime() - t0;

#ifdef DEBUGMATRIX
			imprimir_matriz(seqA, seqB, matriz, tamA, tamB);
#endif

			printf("\nScore: %d\n", pontuacao);

			t0 = MPI_Wtime();
			liberar_matriz(matriz);
			free(seqA);
			free(seqB);
			double t_liberacao = MPI_Wtime() - t0;

			double t_total = MPI_Wtime() - t_inicio_total;
			if (medir_tempos) {
				fprintf(stderr,
						"TEMPOS leitura=%.6f alocacao=%.6f inicializacao=%.6f calculo=%.6f liberacao=%.6f total=%.6f\n",
						t_leitura, t_alocacao, t_inicializacao, t_calculo, t_liberacao, t_total);
			}
		}
	} else if (strcmp(versao, "mpi_blocos") == 0 || strcmp(versao, "hibrido") == 0) {
		MPI_Barrier(MPI_COMM_WORLD);
		double t_inicio_total = MPI_Wtime();
		double t0;

		t0 = MPI_Wtime();
		char *seqA = NULL;
		char *seqB = NULL;
		int tamA = 0;
		int tamB = 0;
		if (meu_ranque == 0) {
			seqA = ler_sequencia(arqA);
			seqB = ler_sequencia(arqB);
			tamA = (int)strlen(seqA);
			tamB = (int)strlen(seqB);
		}
		MPI_Bcast(&tamA, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&tamB, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (meu_ranque > 0) {
			if (tamA > 0) {
				seqA = (char *)malloc((size_t)tamA + 1);
				if (seqA == NULL) {
					fprintf(stderr, "Erro ao alocar sequencia A no ranque %d.\n", meu_ranque);
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
			}
			if (tamB > 0) {
				seqB = (char *)malloc((size_t)tamB + 1);
				if (seqB == NULL) {
					fprintf(stderr, "Erro ao alocar sequencia B no ranque %d.\n", meu_ranque);
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
			}
		}
		if (tamA > 0) {
			MPI_Bcast(seqA, tamA + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
		}
		if (tamB > 0) {
			MPI_Bcast(seqB, tamB + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		double t_leitura = MPI_Wtime() - t0;

		int procs_ativos = (tamB < num_procs) ? tamB : num_procs;
		int base = (procs_ativos > 0) ? (tamB / procs_ativos) : 0;
		int resto = (procs_ativos > 0) ? (tamB % procs_ativos) : 0;
		int linhas_locais = 0;
		if (tamA > 0 && tamB > 0 && meu_ranque < procs_ativos) {
			linhas_locais = base + (meu_ranque < resto ? 1 : 0);
		}

		t0 = MPI_Wtime();
		tipo_matriz **matriz_local = NULL;
		if (tamA > 0 && tamB > 0 && meu_ranque < procs_ativos) {
			matriz_local = alocar_matriz(tamA, linhas_locais);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		double t_alocacao = MPI_Wtime() - t0;

		t0 = MPI_Wtime();
		if (matriz_local != NULL) {
			for (int i = 0; i <= linhas_locais; i++) {
				matriz_local[i][0] = 0;
			}
			if (meu_ranque == 0) {
				for (int j = 0; j <= tamA; j++) {
					matriz_local[0][j] = 0;
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		double t_inicializacao = MPI_Wtime() - t0;

		t0 = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		int pontuacao = 0;
		if (strcmp(versao, "mpi_blocos") == 0) {
			pontuacao = lcs_mpi_blocos(matriz_local, tamA, tamB, seqA, seqB,
					tam_bloco, meu_ranque, num_procs);
		} else {
			pontuacao = lcs_hibrido(matriz_local, tamA, tamB, seqA, seqB,
					tam_bloco, meu_ranque, num_procs);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		double t_calculo = MPI_Wtime() - t0;

#ifdef DEBUGMATRIX
		if (tamA == 0 || tamB == 0) {
			if (meu_ranque == 0) {
				tipo_matriz **matriz_completa = alocar_matriz(tamA, tamB);
				inicializar_matriz(matriz_completa, tamA, tamB);
				imprimir_matriz(seqA, seqB, matriz_completa, tamA, tamB);
				liberar_matriz(matriz_completa);
			}
		} else {
			if (meu_ranque == 0) {
				tipo_matriz **matriz_completa = alocar_matriz(tamA, tamB);
				inicializar_matriz(matriz_completa, tamA, tamB);
				memcpy(&matriz_completa[1][0], &matriz_local[1][0],
						(size_t)linhas_locais * (size_t)(tamA + 1) * sizeof(tipo_matriz));
				for (int r = 1; r < procs_ativos; r++) {
					int linhas_r = base + (r < resto ? 1 : 0);
					int inicio_r = r * base + (r < resto ? r : resto);
					MPI_Recv(&matriz_completa[inicio_r + 1][0], linhas_r * (tamA + 1),
							MPI_UNSIGNED_SHORT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				imprimir_matriz(seqA, seqB, matriz_completa, tamA, tamB);
				liberar_matriz(matriz_completa);
			} else if (meu_ranque < procs_ativos) {
				MPI_Send(&matriz_local[1][0], linhas_locais * (tamA + 1),
						MPI_UNSIGNED_SHORT, 0, 1, MPI_COMM_WORLD);
			}
		}
#endif

		if (meu_ranque == 0) {
			printf("\nScore: %d\n", pontuacao);
		}

		t0 = MPI_Wtime();
		liberar_matriz(matriz_local);
		free(seqA);
		free(seqB);
		MPI_Barrier(MPI_COMM_WORLD);
		double t_liberacao = MPI_Wtime() - t0;

		double t_total = MPI_Wtime() - t_inicio_total;
		if (medir_tempos && meu_ranque == 0) {
			fprintf(stderr,
					"TEMPOS leitura=%.6f alocacao=%.6f inicializacao=%.6f calculo=%.6f liberacao=%.6f total=%.6f\n",
					t_leitura, t_alocacao, t_inicializacao, t_calculo, t_liberacao, t_total);
		}
	} else {
		if (meu_ranque == 0) {
			fprintf(stderr, "Versao desconhecida: %s\n", versao);
		}
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
