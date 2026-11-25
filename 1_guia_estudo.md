# Guia para a defesa do Trabalho 1 (LCS com OpenMP e MPI)

Ler uma vez com calma (30 a 40 min) com o `lcs_paralelo.c` aberto ao lado. Depois disso, só flashcards.

O professor só tem o `lcs_paralelo.c` e o `relatorio_lcs.pdf`. Tudo que for dito tem que bater com esses dois arquivos. O enunciado diz que a nota considera **domínio do tema, robustez da solução e rigor da metodologia**.

---

## 0. Cola rápida (os números que valem decorar)

| O quê | Valor |
|---|---|
| Versão vencedora | `omp_tarefas`, blocos 1024 x 1024 |
| Maior N | 50.000 (matriz de 4,7 GB). N = 52.000 já usava swap |
| Tempo do seq em N = 50k | 6,5 s (abaixo dos 10 s sugeridos, por causa da RAM) |
| Parte sequencial (fs) | 4,04% (quase tudo é o `free`, 3,23%, e a inicialização, 0,79%) |
| Amdahl com 12 / infinito | 8,31 / 24,76 |
| Speedup medido, 12 threads, N = 50k | 2,92 (sobre T1), eficiência 0,24 |
| Speedup sobre o seq, 12 threads, N = 50k | 4,76 |
| Blocos com 1 thread vs seq | 1,6x mais rápido (efeito de cache, não é paralelismo) |
| MPI com 10 processos | speedup 2,20 a 2,33, eficiência 0,22 a 0,23 |
| Escalabilidade fraca | N = 14.000 x raiz(p). Tempo foi de 0,31 s para 1,31 s, eficiência 0,24 |
| Conclusão | Não foi escalável (nem forte nem fraca) nesta máquina |
| Máquina | i7-1265U: 2 P-cores com SMT + 8 E-cores, 12 CPUs lógicas, 15 GiB |

---

## 1. O problema e a dependência (o coração de tudo)

- LCS = maior subsequência comum (não precisa ser contígua). Exemplo do enunciado: `heagawghee` e `pawheae`, resposta 5.
- Programação dinâmica: matriz `(tamB+1) x (tamA+1)`. Linha 0 e coluna 0 valem 0.
- Regra de cada célula:
  - se `A[j-1] == B[i-1]`: `M[i][j] = M[i-1][j-1] + 1` (diagonal + 1)
  - senão: `M[i][j] = max(M[i-1][j], M[i][j-1])` (o maior entre cima e esquerda)
- Resposta: `M[tamB][tamA]`. Custo O(N x M) em tempo e memória.
- **Por que não dá para fazer `parallel for` direto:** cada célula depende da de cima, da esquerda e da diagonal. Os dois laços carregam dependência.
- **Antidiagonal:** a célula (i, j) está na diagonal d = i + j. Ela depende de (i-1, j) e (i, j-1), que estão em d-1, e de (i-1, j-1), que está em d-2. Ou seja, só depende de diagonais anteriores. Células da mesma antidiagonal são independentes e podem ser calculadas ao mesmo tempo. Isso é a **frente de onda (wavefront)**.
- Em blocos é a mesma ideia: o bloco (bi, bj) depende do bloco de cima, do bloco da esquerda e do bloco da diagonal (este último já vem garantido de forma transitiva, porque o de cima depende dele).
- Paralelismo disponível: começa com 1 bloco, cresce até o meio da matriz e depois diminui de novo. No começo e no fim, threads ficam paradas.

---

## 2. Teoria dos slides que foi usada

**Speedup:** S(p) = T(1) / T(p). Compare sempre com o mesmo tamanho de entrada. O slide de "armadilhas" diz que comparar com outro algoritmo mais lento é enganoso. O justo é comparar com o mesmo algoritmo com 1 processador, ou com o melhor algoritmo sequencial. **Por isso usamos T1 da `omp_tarefas` como base**, e mostramos também o seq.

**Eficiência:** E(p) = S(p) / p. Mostra o quanto cada processador está sendo aproveitado.

**Speedup superlinear:** S(p) > p. Pode aparecer por causa de cache (o problema dividido cabe na cache) ou memória. No nosso caso, se a base fosse o seq, teríamos eficiência 1,32 com 2 threads. Esse superlinear seria falso, porque viria da cache dos blocos e não do paralelismo.

**Lei de Amdahl:** S(p) = 1 / (fs + (1 - fs) / p), onde fs é a fração puramente sequencial. Limite com p infinito: 1 / fs. Limitações (slide): ignora comunicação e sincronização, e supõe que fs não muda com o tamanho do problema.

**Lei de Gustafson-Barsis:** se o problema cresce junto com p, a parte paralela cresce e o speedup escalado pode passar o limite de Amdahl. S(p) = p - alfa (p - 1). Tem a ver com escalabilidade fraca. Não usamos no relatório, mas pode ser perguntado.

**Escalabilidade forte:** N fixo, aumenta p, e a eficiência se mantém. Os slides dizem que é difícil de conseguir.

**Escalabilidade fraca:** a eficiência se mantém quando N cresce junto com p. Escalar N com a complexidade: o LCS é O(N²), então N cresce com raiz(p).

**Fontes de overhead (slide):** processos inativos (desbalanceamento, sincronização), comunicação, trabalho extra, caminho crítico.

**Granularidade:** fina = muitas tarefas pequenas (o custo de criar e sincronizar pode passar o ganho). Grossa = poucas tarefas grandes (menos overhead, mas pode faltar paralelismo). O `omp_diag` é o exemplo de granularidade fina demais. Os blocos de 1024 são o meio-termo.

**OpenMP usado:**
- `parallel for` com `schedule(static)`: divide as iterações em partes iguais. Bom quando todas as iterações custam o mesmo.
- `schedule(dynamic, 1)`: cada thread pega uma iteração por vez quando fica livre. Bom quando o custo varia.
- Barreira implícita no fim de `parallel for`, `single` e `parallel`.
- `single`: só uma thread executa o bloco.
- `task`: unidade de trabalho que vai para uma fila e é executada por qualquer thread do time.
- `depend(in: x)` / `depend(out: x)`: a task com `in: x` só começa depois que a task anterior com `out: x` terminou. A ordem vale entre tasks irmãs, na ordem em que foram criadas.
- `firstprivate`: cada task recebe uma cópia da variável com o valor do momento da criação.
- `OMP_PROC_BIND` e `OMP_PLACES`: fixam threads em núcleos (afinidade).

**MPI usado:**
- `MPI_Send` / `MPI_Recv`: ponto a ponto, bloqueantes. O `MPI_Send` padrão pode usar buffer (mensagens pequenas voltam logo) ou esperar o `Recv`.
- `MPI_Bcast`: o rank 0 manda o mesmo dado para todos.
- `MPI_Barrier`: todos esperam todos.
- `MPI_Wtime`: relógio de parede.
- `MPI_Init_thread` com `MPI_THREAD_FUNNELED`: o programa tem threads, mas só a thread principal faz chamadas MPI.

---

## 3. Mapa do código (`lcs_paralelo.c`, 598 linhas)

| Linha | O quê | Por quê |
|---|---|---|
| 11 | `typedef unsigned short tipo_matriz` | Igual ao original (`mtype`). 2 bytes por célula: N = 50k cabe em 4,7 GB. Com `int` seriam 9,3 GB. Limite de 65535, e o Score nunca passa de min(N, M) |
| 13 | `ler_sequencia` | Lê o arquivo ignorando `\n`. Usa `int c` com `fgetc`, para comparar com EOF direito |
| 44 | `alocar_matriz` | **Um único bloco contíguo** + vetor de ponteiros para as linhas (mantém o acesso `matriz[i][j]`). Usa `size_t` porque (N+1)² passa do limite de `int` a partir de N = 46.341. O original fazia um `malloc` por linha |
| 64 | `inicializar_matriz` | Zera linha 0 e coluna 0 |
| 73 | `liberar_matriz` | `free(matriz[0])` libera o bloco inteiro, depois o vetor de ponteiros |
| 81 | `imprimir_matriz` | Mesma saída da `printMatrix` original (usada com `-DDEBUGMATRIX`) |
| 106 | `lcs_sequencial` | O kernel original, sem mudança |
| 123 | `lcs_omp_diagonal` | Laço em `d` (antidiagonal) e `parallel for schedule(static)` (linha 133) nas células da diagonal |
| 146 | `processar_bloco` | Calcula um retângulo da matriz linha a linha. Usado por `omp_blocos`, `omp_tarefas` e `hibrido` |
| 160 | `lcs_omp_blocos` | Antidiagonais de **blocos** (laço `db`, linha 169), `parallel for schedule(dynamic, 1)` (linha 173) |
| 186 | `lcs_omp_tarefas` | **Versão vencedora.** `parallel` + `single` (linhas 202 e 204), uma `task` por bloco com `depend` (linhas 208 a 210) |
| 227 | `lcs_mpi_blocos` | Faixas de linhas por processo e pipeline por blocos de colunas (laço na linha 245, `Recv` na 250, `Send` na 266) |
| 284 | `lcs_hibrido` | Mesmo pipeline do MPI com blocos de colunas mais largos (linha 302). Dentro de cada processo, wavefront de sub-blocos com `parallel for` (linha 329) |
| 360 | `main` | `MPI_Init_thread FUNNELED` (362), versão padrão (370), leitura das flags, dois ramos: OpenMP (402) e MPI (455) |

### 3.1 `omp_tarefas` em detalhe (linhas 186 a 225)

- `num_bi`, `num_bj` = quantidade de blocos em cada direção (arredonda para cima).
- `dep` é um vetor `char` de tamanho `(num_bi+1) x (num_bj+1)`, criado com `calloc`. **O conteúdo nunca é lido nem escrito.** Só os endereços importam: o OpenMP usa o endereço para montar o grafo de dependências.
- É "1-based": o bloco (bi, bj) usa a posição `[bi+1][bj+1]`. A linha 0 e a coluna 0 de `dep` são sentinelas que nenhuma task escreve. Assim os blocos da primeira linha e da primeira coluna não precisam de `if` especial.
- Cada task: `depend(in: dep[bi][bj+1])` (bloco de cima), `depend(in: dep[bi+1][bj])` (bloco da esquerda), `depend(out: dep[bi+1][bj+1])` (ela mesma).
- `single`: só uma thread cria as tasks. As outras já vão executando. Sem o `single`, cada thread criaria todas as tasks de novo.
- As tasks são criadas linha por linha (bi, depois bj), então quem produz sempre é criado antes de quem consome. Isso é necessário para o `depend` funcionar.
- Barreira implícita no fim do `single`/`parallel`: garante que todas as tasks acabaram antes do `free(dep)`.
- **Vantagem sobre `omp_blocos`:** não existe barreira entre diagonais. Um bloco começa assim que os vizinhos de cima e da esquerda terminam, sem esperar a diagonal toda. Isso ajuda muito com núcleos diferentes (um bloco num E-core demora mais e, com barreira, todo mundo esperaria por ele).

### 3.2 `mpi_blocos` em detalhe (linhas 227 a 282)

- **Divisão:** cada processo fica com uma faixa de linhas consecutivas (linhas de B). `base = tamB / procs`, e os primeiros `resto` processos ganham uma linha a mais. `linha_inicio` diz onde começa a faixa.
- `procs_ativos = min(tamB, num_procs)`: se houver mais processos do que linhas, os que sobram não fazem nada.
- **Matriz local:** `alocar_matriz(tamA, linhas_locais)`, ou seja, `linhas_locais + 1` linhas. A **linha 0 local é a borda** que vem do processo anterior (no rank 0 ela é zero).
- **Pipeline:** as colunas são divididas em pedaços de `tam_bloco` (1024). Para cada pedaço:
  1. rank > 0 recebe (`MPI_Recv`) o trecho da última linha do rank anterior e coloca na própria linha 0;
  2. calcula esse pedaço em todas as suas linhas;
  3. envia (`MPI_Send`) o trecho da própria última linha para o próximo rank.
- Assim o rank 0 trabalha no pedaço 2 enquanto o rank 1 trabalha no pedaço 1, e assim por diante. É um wavefront entre processos.
- A **diagonal na borda do pedaço** funciona porque a posição `[0][j_ini-1]` chegou no pedaço anterior e continua na memória.
- `tag` = número do pedaço.
- **Sem deadlock:** a comunicação é uma corrente (0 envia para 1, 1 para 2...). Não existe ciclo, e cada mensagem tem só 2 KB (1024 x 2 bytes), então o `Send` normalmente volta logo (buffer).
- **Score:** o último processo tem a célula final e envia para o rank 0, que imprime.
- `MPI_UNSIGNED_SHORT` é o tipo certo para `tipo_matriz`.

### 3.3 `hibrido` em detalhe (linhas 284 a 358)

- Mesmo pipeline e mesma divisão em faixas do `mpi_blocos`.
- Com mais de 1 thread, o pedaço de colunas do MPI fica 4 vezes mais largo (`tam_bloco * 4` = 4096, linha 302). O motivo é ter vários sub-blocos para as threads dentro de cada pedaço.
- Dentro do pedaço: sub-blocos de 1024 x 1024 em antidiagonais com `parallel for schedule(dynamic, 1)`, igual ao `omp_blocos`.
- Se a faixa ou o pedaço forem estreitos demais (linha 318), calcula sem threads.
- `seqB + linha_inicio`: desloca o ponteiro para que `processar_bloco` use índices locais.
- Só a thread principal chama MPI (fora das regiões paralelas), por isso `FUNNELED` basta.
- Com `OMP_NUM_THREADS=1` ele fica igual ao `mpi_blocos`.

### 3.4 `main` (linhas 360 a 598)

- Sem `-v`: `omp_tarefas` com 1 processo e `hibrido` com vários (linha 370). O híbrido é o padrão com `mpirun` porque o trabalho pede OpenMP + MPI e é a versão que usa os dois.
- Flags: `-v` versão, `-b` tamanho do bloco (padrão 1024), `-a`/`-B` arquivos (padrão `fileA.in`/`fileB.in`), `-m` tempos por fase no stderr.
- **Ramo OpenMP (402):** só o rank 0 trabalha. Mede cada fase com `MPI_Wtime`.
- **Ramo MPI (455):** rank 0 lê os arquivos, `MPI_Bcast` dos tamanhos e das sequências, cada processo aloca só a sua faixa. Há `MPI_Barrier` entre as fases, para o tempo medido ser o do processo mais lento.
- **DEBUGMATRIX no MPI (542):** o rank 0 monta a matriz completa recebendo as faixas dos outros processos (`MPI_Recv` de `linhas * (tamA+1)` valores de uma vez, o que só funciona porque a matriz é contígua) e imprime.
- A saída é `\nScore: %d\n`, igual ao original.

---

## 4. Decisões e porquês (resumo para responder rápido)

- **Tudo em um arquivo com `-v`:** dá para comparar as versões com o mesmo código de leitura, alocação e medição.
- **Blocos em vez de célula a célula:** o `omp_diag` tem cerca de 2N diagonais, cada uma com uma barreira e pouco trabalho (granularidade fina demais). Ficou **mais lento que o seq** (0,52x em N = 10k).
- **Bloco de 1024:** testamos 128, 256, 512 e 1024 (N = 50k, 10 threads). Cálculo: 1,75 s, 1,39 s, 1,09 s, 1,08 s. Blocos pequenos geram tasks demais. Com 1024, uma linha do bloco tem 2 KB, e as duas linhas usadas (atual e anterior) cabem na L1 (48 KB no P-core, 32 KB no E-core).
- **Tasks em vez de `parallel for` em blocos:** sem barreira entre diagonais (seção 3.1).
- **`dynamic, 1` no `omp_blocos`:** blocos da borda são menores e os núcleos têm velocidades diferentes.
- **Matriz contígua:** um `malloc` só, `free` mais simples e permite mandar várias linhas numa mensagem MPI.
- **Matriz inteira na memória:** o enunciado exige (não dá para guardar só 2 linhas). É isso que limita o N.
- **MPI com faixas de linhas:** cada processo só precisa da última linha do anterior, então a comunicação é pequena (uma borda por pedaço).

---

## 5. Metodologia (item 3 do relatório)

- Notebook pessoal, sem VM. i7-1265U: 2 P-cores (SMT, CPUs 0-1 e 2-3, até 4,8 GHz) e 8 E-cores (CPUs 4 a 11, até 3,6 GHz). L3 12 MB, 15 GiB RAM.
- Ubuntu 24.04.5, kernel 7.0.0-43, gcc 13.3.0 via mpicc (Open MPI 4.1.6), `-O3 -fopenmp -Wall`.
- **Não** desligamos o turbo, deixamos o governor `powersave` (padrão), não isolamos núcleos, e a IDE ficou aberta. O relatório admite isso e mostra o efeito na variação.
- **Afinidade:** `taskset`, `OMP_PROC_BIND=true` e `OMP_PLACES`. Ordem: P-cores primeiro. 1 = CPU 0; 2 = CPUs 0 e 2 (os dois P-cores, sem SMT); 4 = 2P + 2E; ...; 10 = todos os físicos; 12 = todas as lógicas (com SMT). No MPI: `mpirun --bind-to none` + `taskset` por processo.
- Entradas: DNA aleatório (A, C, G, T) com semente fixa. N = 15k, 30k, 50k.
- 20 execuções por configuração. Média e desvio padrão amostral. Speedup e eficiência calculados sobre as médias.
- Tempo: medido dentro do programa com `MPI_Wtime`, da leitura até o `free`.
- **Corretude:** comparamos o Score **e a matriz inteira** (`-DDEBUGMATRIX`) com o `lcs.c` original. Casos: exemplo do enunciado, sequências vazias, 1 caractere, tamanhos que não são múltiplos do bloco (primos como 101 x 97), retangulares (5 x 500), mais processos do que linhas, 1 a 8 threads (inclusive ímpares), 1 a 5 processos, e repetições para pegar condição de corrida. O enunciado avisa que "o resultado bater não significa que está correto", por isso comparamos a matriz toda e não só o Score.

---

## 6. Resultados (itens 4 a 9)

**Região sequencial (item 4):** seq com N = 50k, por fase. Leitura 0,02%, alocação 0,00%, inicialização 0,79%, **cálculo 95,96%**, liberação 3,23%. Total fora do kernel: **4,04%**. A inicialização pesa mais do que parece: cada linha tem 100 KB, então escrever a coluna 0 toca uma página diferente por linha e causa uma falta de página em cada uma.

**Amdahl (item 5):** fs = 0,0404. p = 2: 1,92; 4: 3,57; 8: 6,24; 12: 8,31; infinito: 24,76.

**Speedup e eficiência (itens 6 e 7), omp_tarefas, base T1:**
- N = 50k: speedup 1,62 (2), 1,99 (4), 2,64 (8), 2,92 (12). Eficiência 0,81 (2), 0,50 (4), 0,24 (12).
- Maior ganho de 1 para 2 threads (os dois P-cores). De 2 para 4 o ganho cai porque entram E-cores, que são mais lentos.
- Medido (2,92) bem abaixo de Amdahl (8,31). **Por quê:**
  1. O LCS faz pouquíssima conta por célula (uma comparação e um max ou uma soma) para cada 2 bytes escritos. Depende mais da memória do que da CPU, e a banda de memória é dividida.
  2. Amdahl supõe núcleos iguais, mas 8 dos 10 são E-cores.
  3. Começo e fim da frente de onda com pouco paralelismo.
- Eficiência sobe um pouco com N (10 threads: 0,25, 0,27, 0,28), porque há mais blocos (225 em 15k, 2.401 em 50k) e o começo e o fim da onda pesam menos.
- **MPI:** speedup 2,20 a 2,33 com 10 processos, contra 2,52 a 2,81 do OpenMP com 10 threads. Custo extra do pipeline: o último processo só começa depois de P - 1 pedaços. Em N = 15k (15 pedaços, 10 processos), todos trabalham juntos só em 15 / 24 = 62% do tempo. Em N = 50k (49 pedaços), 84%. Mas a eficiência quase não mudou com N, então a memória pesa mais.

**Escalabilidade (item 8):** não foi escalável. Forte: com N fixo a eficiência cai de 1 para cerca de 0,24. Fraca: com N = 14.000 x raiz(p), o tempo deveria ficar em 0,31 s, mas subiu até 1,31 s (eficiência 0,24). Motivos: a matriz cresce de 392 MB para 4,7 GB com a mesma banda de memória, as threads novas são E-cores ou SMT, e a parte sequencial (inicialização e free) cresce com a matriz (0,04 s para 0,39 s).

**Análise (item 9):**
- **Coerente:** mesmo Score em tudo, tempo cai com mais threads, speedup abaixo de Amdahl. O "estranho" (1 thread mais rápida que o seq) é cache.
- **SMT:** 4 threads em 2 P-cores com SMT foram 2,7%, 11,0% e 12,9% mais lentas que 4 threads em núcleos físicos separados. Duas threads no mesmo núcleo dividem unidades de execução e caches L1/L2, e o programa depende de memória. De 10 (sem SMT) para 12 (com SMT) o ganho foi só de 1,8%, 5,5% e 3,9%.
- **P vs E:** 8 threads com 2P + 6E foram 11,5% mais rápidas no total do que 8 E-cores. No cálculo, só 4,9% (as tasks se distribuem e os núcleos rápidos pegam mais blocos). O resto vem da parte sequencial, que roda na thread 0 (P-core num caso, E-core no outro).
- **Desvio padrão:** o seq em 50k teve 7,5%. As 15 primeiras execuções deram cerca de 6,05 s e as 5 últimas 6,98 s: provável queda de frequência por temperatura depois de ~1,5 min, já que o turbo estava ligado. Em entradas pequenas (menos de 1 s), desvio de 9% a 17%. Em 50k, de 1,6% a 4,3%. O MPI varia mais (um atraso se propaga pelo pipeline). A bateria MPI separada (programas fechados) teve desvio de 0,7% a 5,3%, mas os tempos absolutos ficaram ~18% diferentes da bateria anterior. Por isso só comparamos speedups dentro da mesma bateria.

---

## 7. Limitações (saber admitir sem se enrolar)

- N = 50k dá 6,5 s no seq, abaixo dos 10 s sugeridos. Não dava para aumentar sem swap, porque a matriz inteira tem que ficar na memória.
- Turbo ligado, governor `powersave`, IDE aberta. Desligar o turbo e fixar o governor em `performance` reduziria a variação.
- O híbrido só foi medido na Tabela 1 (melhor configuração 5 processos x 2 threads), não teve tabela de escalabilidade.
- MPI e híbrido numa máquina só não têm vantagem. Fariam sentido em várias máquinas (memória somada maior, N maior).
- "Limitado por memória" é nossa explicação, baseada nos resultados (pouca conta por célula, SMT ajuda pouco, eficiência baixa). Não medimos com ferramenta (ex.: `perf`). Outra causa provável: as páginas da matriz são tocadas pela primeira vez durante o cálculo, então as faltas de página acontecem dentro do kernel paralelo.
- `MPI_Init_thread` não confere o nível de suporte retornado (`(void)suporte`). Funciona no Open MPI usado, mas o certo seria checar.
- O híbrido usa `parallel for` por diagonal dentro do processo (com barreira), e não tasks.

## 8. Dificuldades e o que aprendemos

- **Dependência:** entender que a paralelização tinha que ser por antidiagonais (wavefront).
- **Granularidade:** a primeira versão (`omp_diag`) ficou mais lenta que o seq. Aprendemos que barreira e pouco trabalho por região matam o ganho. Daí os blocos.
- **Cache:** a versão em blocos com 1 thread já ficou 1,6x mais rápida. Isso nos obrigou a pensar em qual referência usar para o speedup.
- **Barreiras:** `omp_blocos` -> `omp_tarefas` tirou a barreira entre diagonais.
- **Memória:** a matriz inteira limita o N. Precisamos do `unsigned short` e do `size_t` (overflow de `int` a partir de N = 46.341).
- **Máquina heterogênea:** P-cores e E-cores mudam tudo. Tivemos que pensar na ordem de afinidade e explicar o efeito do SMT.
- **Medição:** temperatura e programas abertos mudam os tempos. Medir 20 vezes e olhar o desvio mostrou isso.
- **MPI:** pensar em borda (linha 0 local), pipeline e tags. Ver na prática que numa máquina só o MPI não ajuda muito.

---

## 9. Divisão sugerida da fala (os dois precisam saber tudo)

- **Felipe:** problema e dependência, estratégia (por que blocos e tasks), código (mostrar `lcs_omp_tarefas` e `lcs_mpi_blocos`).
- **Victor:** metodologia, região sequencial e Amdahl, tabelas de speedup e eficiência, escalabilidade, SMT e desvio padrão.

Quem não está falando fica com o código aberto, pronto para mostrar a linha certa.
