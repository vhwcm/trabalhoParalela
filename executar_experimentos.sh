#!/usr/bin/env bash
set -euo pipefail

if [ -d "/usr/lib64/openmpi/bin" ]; then
    export PATH="/usr/lib64/openmpi/bin:$PATH"
fi

MODO="${1:---validar}"
REPETICOES=20
ARQUIVO_CSV="resultados_experimentos.csv"
TEM_MPI=0
FLAG_OVERSUBSCRIBE=""

uso() {
    echo "Uso: $0 [--validar | --completo]"
    echo "  --validar   Executa apenas testes de corretude e edge cases."
    echo "  --completo  Executa validacao e benchmarks (20 repeticoes por configuracao)."
}

verificar_ambiente() {
    if [ ! -x ./lcs ] || [ lcs_source_code/lcs.c -nt ./lcs ]; then
        echo "[INFO] Compilando lcs (original)..."
        gcc -O3 -Wall -o lcs lcs_source_code/lcs.c
    fi

    if [ ! -x ./lcs_sequencial ] || [ lcs_sequencial.c -nt ./lcs_sequencial ]; then
        echo "[INFO] Compilando lcs_sequencial..."
        gcc -O3 -Wall -o lcs_sequencial lcs_sequencial.c
    fi

    if command -v mpicc >/dev/null 2>&1 && command -v mpirun >/dev/null 2>&1; then
        TEM_MPI=1
        if [ ! -x ./lcs_paralelo ] || [ lcs_paralelo.c -nt ./lcs_paralelo ]; then
            echo "[INFO] Compilando lcs_paralelo..."
            mpicc -O3 -fopenmp -Wall -o lcs_paralelo lcs_paralelo.c
        fi
        if mpirun --help 2>&1 | grep -q -- "--oversubscribe"; then
            FLAG_OVERSUBSCRIBE="--oversubscribe"
        fi
    else
        TEM_MPI=0
        echo "[AVISO] mpicc/mpirun nao encontrados no PATH."
        echo "[AVISO] Apenas lcs_sequencial sera validado/executado; parte paralela aguarda instalacao do OpenMPI."
    fi
}

escrever_entradas_literais() {
    local texto_a="$1"
    local texto_b="$2"
    printf "%s" "$texto_a" > fileA.in
    printf "%s" "$texto_b" > fileB.in
}

validar_caso() {
    local nome="$1"
    local esperado="$2"

    local tmp_orig tmp_seq tmp_par
    tmp_orig="$(mktemp)"
    tmp_seq="$(mktemp)"
    tmp_par="$(mktemp)"

    ./lcs > "$tmp_orig"
    ./lcs_sequencial > "$tmp_seq"

    if ! cmp -s "$tmp_orig" "$tmp_seq"; then
        echo "[ERRO] Caso '$nome': saida de lcs_sequencial difere de lcs (original)."
        echo "--- lcs ---"
        cat "$tmp_orig"
        echo "--- lcs_sequencial ---"
        cat "$tmp_seq"
        rm -f "$tmp_orig" "$tmp_seq" "$tmp_par"
        exit 1
    fi

    local score_seq
    score_seq="$(awk '/Score:/ {print $2}' "$tmp_seq" | tr -d '[:space:]')"

    if [ -n "$esperado" ] && [ "$score_seq" != "$esperado" ]; then
        echo "[ERRO] Caso '$nome': lcs_sequencial retornou Score='$score_seq', esperado '$esperado'."
        rm -f "$tmp_orig" "$tmp_seq" "$tmp_par"
        exit 1
    fi

    if [ "$TEM_MPI" -eq 1 ]; then
        for p in 1 2 3 4; do
            for t in 1 2 3 4; do
                OMP_NUM_THREADS="$t" mpirun $FLAG_OVERSUBSCRIBE -np "$p" ./lcs_paralelo > "$tmp_par"
                if ! cmp -s "$tmp_seq" "$tmp_par"; then
                    echo "[ERRO] Caso '$nome' (P=$p, T=$t): saida de lcs_paralelo difere de lcs_sequencial."
                    echo "--- lcs_sequencial ---"
                    cat "$tmp_seq"
                    echo "--- lcs_paralelo ---"
                    cat "$tmp_par"
                    rm -f "$tmp_orig" "$tmp_seq" "$tmp_par"
                    exit 1
                fi
            done
        done
        echo "[OK] $nome -> Score=$score_seq (lcs, lcs_sequencial e lcs_paralelo P=1..4, T=1..4 identicos)"
    else
        echo "[OK] $nome -> Score=$score_seq (lcs e lcs_sequencial identicos)"
    fi

    rm -f "$tmp_orig" "$tmp_seq" "$tmp_par"
}

executar_validacao() {
    echo "=== Validacao de Corretude e Edge Cases ==="

    # 1. Strings de tamanho 1 (iguais e diferentes)
    escrever_entradas_literais "a" "a"
    validar_caso "tam_1_iguais" "1"

    escrever_entradas_literais "a" "b"
    validar_caso "tam_1_diferentes" "0"

    # 2. Exemplo original de lcs_source_code/fileA.in e fileB.in
    cp lcs_source_code/fileA.in fileA.in
    cp lcs_source_code/fileB.in fileB.in
    validar_caso "exemplo_original_fileA_fileB" "5"

    # 3. Strings identicas
    python3 gerar_entradas.py 100 100 iguais --semente 42
    validar_caso "strings_identicas_100" "100"

    # 4. Strings totalmente diferentes (LCS = 0)
    python3 gerar_entradas.py 100 100 diferentes --semente 42
    validar_caso "strings_diferentes_100" "0"

    # 5. Tamanhos desiguais
    python3 gerar_entradas.py 10 500 aleatorio --semente 42
    validar_caso "tamanhos_desiguais_10_500" ""

    python3 gerar_entradas.py 500 10 aleatorio --semente 42
    validar_caso "tamanhos_desiguais_500_10" ""

    # 6. Caracteres alfanumericos mistos
    python3 gerar_entradas.py 200 200 aleatorio --semente 99
    validar_caso "alfanumericos_200" ""

    echo "=== Validacao concluida com sucesso ==="
}

calcular_estatisticas_csv() {
    local arquivo_tempos="$1"
    python3 - "$arquivo_tempos" <<'EOF'
import math
import sys

caminho = sys.argv[1]
valores = []
with open(caminho, "r", encoding="utf-8") as f:
    for linha in f:
        linha = linha.strip()
        if linha:
            valores.append(float(linha))

n = len(valores)
media = sum(valores) / n
if n > 1:
    var = sum((x - media) ** 2 for x in valores) / (n - 1)
    desvio = math.sqrt(var)
else:
    desvio = 0.0
print(f"{media:.6f},{desvio:.6f}")
EOF
}

medir_sequencial_20x() {
    local experimento="$1"
    local tam="$2"

    echo "[BENCH] $experimento (N=$tam, ${REPETICOES}x)..."
    python3 gerar_entradas.py "$tam" "$tam" aleatorio --semente 42

    local tmp_tot tmp_seq tmp_ker
    tmp_tot="$(mktemp)"
    tmp_seq="$(mktemp)"
    tmp_ker="$(mktemp)"

    for _ in $(seq 1 "$REPETICOES"); do
        local saida_err
        saida_err="$(./lcs_sequencial -t 2>&1 >/dev/null)"
        echo "$saida_err" | awk '/Tempo total:/ {print $3}' >> "$tmp_tot"
        echo "$saida_err" | awk '/Tempo sequencial:/ {print $3}' >> "$tmp_seq"
        echo "$saida_err" | awk '/Tempo kernel:/ {print $3}' >> "$tmp_ker"
    done

    local est_tot est_seq est_ker fracao_seq
    est_tot="$(calcular_estatisticas_csv "$tmp_tot")"
    est_seq="$(calcular_estatisticas_csv "$tmp_seq")"
    est_ker="$(calcular_estatisticas_csv "$tmp_ker")"

    local media_tot media_seq
    media_tot="$(echo "$est_tot" | cut -d',' -f1)"
    media_seq="$(echo "$est_seq" | cut -d',' -f1)"
    fracao_seq="$(python3 -c "print(f'{($media_seq / $media_tot):.6f}')")"

    echo "${experimento},${tam},${tam},1,1,${REPETICOES},${est_tot},${est_seq},${est_ker},${fracao_seq}" >> "$ARQUIVO_CSV"

    rm -f "$tmp_tot" "$tmp_seq" "$tmp_ker"
}

medir_paralelo_20x() {
    local experimento="$1"
    local tam="$2"
    local procs="$3"
    local threads="$4"

    echo "[BENCH] $experimento (N=$tam, P=$procs, T=$threads, ${REPETICOES}x)..."
    python3 gerar_entradas.py "$tam" "$tam" aleatorio --semente 42

    local tmp_tot
    tmp_tot="$(mktemp)"

    for _ in $(seq 1 "$REPETICOES"); do
        python3 - "$procs" "$threads" "$FLAG_OVERSUBSCRIBE" <<'EOF' >> "$tmp_tot"
import os
import subprocess
import sys
import time

procs = sys.argv[1]
threads = sys.argv[2]
oversub = sys.argv[3]

cmd = ["mpirun"]
if oversub:
    cmd.append(oversub)
cmd.extend(["-np", procs, "./lcs_paralelo"])

env = os.environ.copy()
env["OMP_NUM_THREADS"] = threads

inicio = time.perf_counter()
subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
fim = time.perf_counter()
print(f"{fim - inicio:.6f}")
EOF
    done

    local est_tot
    est_tot="$(calcular_estatisticas_csv "$tmp_tot")"
    echo "${experimento},${tam},${tam},${procs},${threads},${REPETICOES},${est_tot},NA,NA,NA,NA,NA" >> "$ARQUIVO_CSV"

    rm -f "$tmp_tot"
}

executar_benchmarks() {
    echo "=== Iniciando Benchmarks (${REPETICOES} repeticoes por configuracao) ==="
    echo "experimento,tam_a,tam_b,processos,threads,repeticoes,media_total_s,desvio_total_s,media_seq_s,desvio_seq_s,media_kernel_s,desvio_kernel_s,fracao_seq" > "$ARQUIVO_CSV"

    # 1. Sequencial + Fracao Sequencial (Lei de Amdahl)
    for tam in 2000 5000 10000; do
        medir_sequencial_20x "sequencial_amdahl" "$tam"
    done

    if [ "$TEM_MPI" -eq 0 ]; then
        echo "[AVISO] Benchmarks paralelos nao executados pois OpenMPI (mpicc/mpirun) nao esta instalado."
        echo "[INFO] Resultados sequenciais salvos em $ARQUIVO_CSV."
        return
    fi

    # 2. Escalabilidade Forte (N=10000, ajustado para 4 nucleos fisicos / 8 threads logicas e 7.5 GiB RAM)
    local configs_forte=(
        "1 1"
        "1 2"
        "2 1"
        "1 4"
        "2 2"
        "4 1"
        "1 8"
        "2 4"
        "4 2"
    )
    for cfg in "${configs_forte[@]}"; do
        read -r p t <<< "$cfg"
        medir_paralelo_20x "escalabilidade_forte" 10000 "$p" "$t"
    done

    # 3. Escalabilidade Fraca (carga por trabalhador constante: N_base=4000, N = 4000 * sqrt(P*T))
    local configs_fraca=(
        "4000 1 1"
        "5657 1 2"
        "5657 2 1"
        "8000 1 4"
        "8000 2 2"
        "8000 4 1"
        "11314 1 8"
        "11314 2 4"
        "11314 4 2"
    )
    for cfg in "${configs_fraca[@]}"; do
        read -r tam p t <<< "$cfg"
        medir_paralelo_20x "escalabilidade_fraca" "$tam" "$p" "$t"
    done

    echo "=== Benchmarks concluidos. Resultados salvos em $ARQUIVO_CSV ==="
}

case "$MODO" in
    --validar)
        verificar_ambiente
        executar_validacao
        ;;
    --completo)
        verificar_ambiente
        executar_validacao
        executar_benchmarks
        ;;
    -h|--help)
        uso
        ;;
    *)
        echo "[ERRO] Opcao invalida: $MODO"
        uso
        exit 1
        ;;
esac
