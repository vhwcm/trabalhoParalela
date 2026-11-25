# Perguntas difíceis (e como responder)

Responder curto, com o motivo, e mostrar o código quando der. Se não souber, dizer o que acha e por quê, e admitir que não mediu. Isso é melhor do que inventar.

---

**1. Por que o speedup (2,9) ficou tão longe de Amdahl (8,3)?**
Amdahl só considera a parte sequencial. Ele ignora três coisas que pesam aqui: (1) o LCS é limitado por memória (uma comparação e um max por célula, para cada 2 bytes escritos), e a banda é dividida entre os núcleos; (2) Amdahl supõe núcleos iguais, mas 8 dos 10 são E-cores; (3) a frente de onda tem pouco paralelismo no começo e no fim. O slide de limitações de Amdahl diz justamente que ele ignora comunicação e sincronização.

**2. Vocês mediram que é limitado por memória?**
Não diretamente, não usamos `perf`. É a explicação mais coerente com o que medimos: SMT quase não ajuda (às vezes piora), a eficiência é baixa mesmo com N grande, e a conta por célula é mínima. Outra causa provável: a matriz é tocada pela primeira vez durante o cálculo, então as faltas de página acontecem dentro da parte paralela.

**3. Por que usar T1 da omp_tarefas como base e não o seq?**
A versão em blocos com 1 thread já é 1,6x mais rápida que o seq, por causa da cache e não do paralelismo. Com o seq como base, a eficiência com 2 threads daria 1,32: um superlinear falso. O slide de "armadilhas do speedup" diz que o justo é comparar o mesmo algoritmo. Mesmo assim, mostramos o seq na tabela e o speedup sobre ele (4,76 com 12 threads).

**4. Por que a versão em blocos com 1 thread é mais rápida que o seq?**
No seq, cada linha tem 50.000 células (100 KB). Quando o laço volta para ler a linha anterior, ela já saiu da L1. No bloco de 1024, as duas linhas usadas somam 4 KB e ficam na L1.

**5. Mostrem onde está a dependência entre os blocos.**
`lcs_omp_tarefas`, linhas 208 a 210. `depend(in)` no bloco de cima (`dep[bi][bj+1]`) e no da esquerda (`dep[bi+1][bj]`), `depend(out)` no próprio bloco (`dep[bi+1][bj+1]`).

**6. E a dependência da diagonal (bloco de cima à esquerda)?**
Já está garantida de forma transitiva: o bloco de cima depende do bloco da diagonal (é o vizinho da esquerda dele), então quando o de cima termina, o da diagonal já terminou.

**7. Para que serve o vetor `dep`? O que tem dentro?**
Nada útil: é alocado com `calloc` e nunca é lido nem escrito. O OpenMP usa só os endereços para saber quem depende de quem. Tem uma linha e uma coluna a mais (sentinelas) para que os blocos da borda não precisem de `if`.

**8. Por que o `single`? E se tirar?**
Só uma thread cria as tasks, e as outras executam. Sem o `single`, cada thread do time criaria todas as tasks, e cada bloco seria calculado várias vezes.

**9. Como vocês sabem que todas as tasks terminaram antes de retornar o Score?**
Barreira implícita no fim do `single` e do `parallel`. A thread só passa depois que todas as tasks do time terminam.

**10. Por que o `omp_diag` ficou mais lento que o sequencial?**
Granularidade fina demais. São cerca de 2N antidiagonais (20.000 em N = 10k), cada uma com um `parallel for` e uma barreira, e várias diagonais têm poucas células. O custo de sincronização passa o ganho.

**11. Por que blocos de 1024?**
Testamos 128, 256, 512 e 1024 com N = 50k e 10 threads: 1,75 s, 1,39 s, 1,09 s, 1,08 s. Bloco pequeno gera tasks demais (overhead). Com 1024, cada linha do bloco tem 2 KB e as duas linhas em uso cabem na L1. Ainda sobram 2.401 blocos em N = 50k, paralelismo suficiente para 12 threads.

**12. Por que no `omp_blocos` usaram `schedule(dynamic, 1)`?**
Os blocos da borda podem ser menores (N não é múltiplo de 1024) e os núcleos têm velocidades diferentes. Com `dynamic`, quem termina primeiro pega o próximo bloco.

**13. Por que o MPI escalou pior que o OpenMP?**
Numa máquina só, o MPI tem os mesmos limites do OpenMP (memória e E-cores) e ainda paga o pipeline: o último processo só começa depois que os anteriores fizeram P - 1 pedaços. Também copia a borda entre processos, enquanto no OpenMP todos enxergam a mesma matriz. Em N = 15k com 10 processos, todos trabalham juntos só em 62% do tempo.

**14. Então por que o mpi_blocos ganhou em N = 10.000 (Tabela 1)?**
Porque cada processo aloca, inicializa e libera só a própria faixa. Essas fases, que são sequenciais no OpenMP, ficam divididas entre os processos. Em entradas pequenas elas pesam bastante.

**15. Pode dar deadlock no pipeline MPI?**
Não. É uma corrente: cada processo só recebe do anterior e só envia para o próximo, sem ciclo. O rank 0 só envia e o último só recebe. E cada mensagem tem 2 KB, então o `MPI_Send` normalmente volta logo (o MPI copia para um buffer).

**16. Como a diagonal funciona na borda de um pedaço de colunas, no MPI?**
A célula `[1][j_ini]` precisa de `[0][j_ini-1]`. Esse valor chegou no pedaço anterior (ou é zero na coluna 0) e continua na linha 0 da matriz local.

**17. O enunciado exige a matriz inteira em memória. No MPI ela está?**
Sim, mas dividida: cada processo guarda a sua faixa, e juntas elas formam a matriz inteira. Com `-DDEBUGMATRIX`, o rank 0 recebe as faixas e imprime a matriz completa (linha 542 em diante).

**18. Por que o padrão com `mpirun` é o híbrido, se o mpi_blocos foi mais rápido?**
O trabalho pede OpenMP + MPI, e o híbrido é a versão que usa os dois. Ele faz mais sentido em várias máquinas, com um processo por máquina e threads dentro. Com `OMP_NUM_THREADS=1` ele fica igual ao mpi_blocos.

**19. Por que `MPI_THREAD_FUNNELED`?**
No híbrido existem threads OpenMP, mas só a thread principal chama MPI, sempre fora das regiões paralelas. FUNNELED é exatamente esse caso. Ponto fraco: não conferimos o valor de `suporte` retornado.

**20. Por que `unsigned short`? Qual o limite?**
Igual ao original. São 2 bytes por célula, o que deixa N = 50k em 4,7 GB. Com `int` seriam 9,3 GB e não caberia. O limite é 65535, e o Score nunca passa do tamanho da menor sequência, então funciona até N = 65535.

**21. Por que `size_t` na alocação?**
(N+1)² passa do maior `int` a partir de N = 46.341. Com N = 50k, a conta em `int` daria overflow e alocaria o tamanho errado.

**22. Por que a alocação é contígua, diferente do original?**
Um `malloc` só, `free` mais simples, e dá para enviar várias linhas numa mensagem MPI só (usado no DEBUGMATRIX). O acesso continua `matriz[i][j]` por causa do vetor de ponteiros.

**23. Por que o seq em N = 50k levou só 6,5 s se o enunciado sugere 10 s?**
Faltou RAM. A matriz inteira tem que ficar em memória e ocupa 4,7 GB. Com N = 52.000 a máquina começou a usar swap, e isso estragaria as medições.

**24. Vocês desligaram o turbo? Por que não?**
Não. Deixamos o turbo ligado e o governor `powersave`. Vimos o efeito depois: no seq em 50k, as 5 últimas execuções ficaram ~15% mais lentas (provável queda de frequência por temperatura). Se fôssemos refazer, desligaríamos o turbo, fixaríamos o governor em `performance` e fecharíamos os outros programas.

**25. O que o desvio padrão mostra?**
Em execuções curtas (menos de 1 s), o desvio é alto (9% a 17%): subida de frequência, faltas de página e outros programas pesam mais. Em N = 50k fica entre 1,6% e 4,3%. O MPI varia mais porque um atraso num processo atrasa todos os seguintes no pipeline. Entre baterias feitas em momentos diferentes, a diferença chegou a 18%, por isso só comparamos speedups dentro da mesma bateria.

**26. Por que o SMT piorou?**
As duas threads no mesmo núcleo dividem unidades de execução e caches L1 e L2. O programa depende de memória, então a segunda thread ajuda pouco e ainda disputa cache. 4 threads com SMT foram até 12,9% mais lentas que 4 em núcleos separados.

**27. O algoritmo é escalável?**
Não nesta máquina. Forte: com N fixo, a eficiência cai de 1 para ~0,24. Fraca: aumentando N com raiz(p), o tempo deveria ficar constante (0,31 s), mas foi para 1,31 s. A eficiência melhora um pouco com N maior, mas não compensa.

**28. Por que N cresce com raiz(p) na fraca?**
O trabalho é N², então para cada thread ter a mesma quantidade de células, N² tem que crescer com p, ou seja, N com raiz(p).

**29. Como garantiram que o resultado está correto?**
Comparamos o Score e a matriz inteira (`-DDEBUGMATRIX`) com o `lcs.c` original, em todas as versões. Casos de borda: vazias, 1 caractere, tamanhos primos (não múltiplos do bloco), retangulares, mais processos do que linhas, número ímpar de threads, e repetições para pegar condição de corrida. Nos experimentos, o Score também foi igual em todas as execuções.

**30. Por que as threads foram fixadas começando pelos P-cores?**
Para que 1 e 2 threads rodem nos núcleos mais rápidos, sem o sistema migrar threads entre núcleos, o que reduz a variação. Os E-cores só entram a partir de 4 threads, e o SMT só com 12.

**31. Teria como melhorar?**
Desligar turbo e fixar frequência; usar tasks também dentro do híbrido (tirar as barreiras); usar `MPI_Isend`/`MPI_Irecv` para sobrepor comunicação e cálculo; rodar o MPI em várias máquinas para usar N maior.

**32. Vocês usaram IA?**
Respondam com a verdade e de acordo com a regra da disciplina. O mais importante é mostrar que entendem cada parte, e é para isso que serve este material.
