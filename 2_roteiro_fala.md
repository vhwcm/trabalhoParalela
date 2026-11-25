# Roteiro da fala (2 a 3 minutos)

Não decorar palavra por palavra. Decorar a ordem dos blocos e os números em negrito.

---

**[Felipe] 1. Problema (20 s)**
"O LCS calcula o tamanho da maior subsequência comum entre duas sequências usando uma matriz de programação dinâmica. Cada célula depende da de cima, da esquerda e da diagonal, então não dá para paralelizar os laços direto. Mas as células de uma mesma antidiagonal são independentes, e essa é a base de todas as nossas versões: uma frente de onda."

**[Felipe] 2. Estratégia (40 s)**
"Fizemos seis versões no mesmo arquivo, escolhidas pela flag `-v`. A primeira, um parallel for por antidiagonal de células, ficou mais lenta que o sequencial, porque eram cerca de 2N barreiras com pouco trabalho cada. Então passamos a dividir a matriz em blocos de **1024 x 1024**. A vencedora foi a **omp_tarefas**: cada bloco vira uma task do OpenMP que depende do bloco de cima e do bloco da esquerda. Assim um bloco começa assim que os vizinhos terminam, sem barreira entre as diagonais. Também fizemos uma versão MPI, em que cada processo fica com uma faixa de linhas e passa a borda para o próximo em pipeline, e uma híbrida, com OpenMP dentro de cada processo."
(Mostrar `lcs_omp_tarefas`, linha 186, e o `depend` nas linhas 208 a 210.)

**[Victor] 3. Metodologia (30 s)**
"Rodamos num notebook com i7-1265U, que tem **2 P-cores com SMT e 8 E-cores**. Fixamos as threads nos núcleos começando pelos P-cores. Cada configuração rodou **20 vezes**, com média e desvio padrão. Conferimos o Score e a matriz inteira contra o código original. O maior N foi **50.000**, porque a matriz ocupa 4,7 GB e acima disso começava a usar swap. Não desligamos o turbo, e isso aparece na variação."

**[Victor] 4. Resultados (40 s)**
"A parte puramente sequencial foi de **4%**, quase toda do free e da inicialização. Por Amdahl, o speedup máximo com 12 threads seria **8,3**. Medimos **2,9** com 12 threads em N = 50.000, com eficiência de **0,24**. A diferença vem principalmente de três coisas: o LCS faz pouca conta por célula e depende muito da memória, 8 dos 10 núcleos são E-cores mais lentos, e o começo e o fim da frente de onda têm pouco paralelismo. O SMT piorou o desempenho em até 13%. O MPI escalou um pouco pior que o OpenMP, porque numa máquina só ele paga o pipeline e as cópias de borda."

**[Victor ou Felipe] 5. Conclusão (20 s)**
"A eficiência caiu tanto com N fixo quanto na escalabilidade fraca, então nesta máquina o algoritmo não foi escalável. O que mais aprendemos: a granularidade importa muito, a cache pode dar um falso speedup superlinear se a referência for mal escolhida, e numa máquina heterogênea a afinidade muda os resultados."
