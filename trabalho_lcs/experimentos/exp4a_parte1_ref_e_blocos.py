#!/usr/bin/env python3
import csv
import os
import re
import statistics
import subprocess

BIN = "/home/felps/Documents/paralela/trabalho_lcs/lcs_paralelo"
ENTRADAS = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas"
RES_DIR = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/resultados"

RE_SCORE = re.compile(r"Score:\s*(\d+)")
RE_TEMPOS = re.compile(
    r"TEMPOS\s+leitura=([\d.]+)\s+alocacao=([\d.]+)\s+inicializacao=([\d.]+)\s+calculo=([\d.]+)\s+liberacao=([\d.]+)\s+total=([\d.]+)"
)


def rodar_omp(versao: str, n: int, cpus: str, threads: int, bloco: int | None, rep: int):
    arq_a = f"{ENTRADAS}/dna_{n}_A.in"
    arq_b = f"{ENTRADAS}/dna_{n}_B.in"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "true"
    env["OMP_PLACES"] = "threads"
    cmd = ["taskset", "-c", cpus, BIN, "-v", versao]
    if bloco is not None:
        cmd += ["-b", str(bloco)]
    cmd += ["-a", arq_a, "-B", arq_b, "-m"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    saida = proc.stdout + "\n" + proc.stderr
    score = int(RE_SCORE.search(saida).group(1))
    leitura, aloc, init, calc, lib, total = map(float, RE_TEMPOS.search(saida).groups())
    return score, leitura, aloc, init, calc, lib, total


def rodar_mpi(versao: str, n: int, rank_cpus: list[str], threads_per_rank: int, bloco: int, rep: int):
    arq_a = f"{ENTRADAS}/dna_{n}_A.in"
    arq_b = f"{ENTRADAS}/dna_{n}_B.in"
    p = len(rank_cpus)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads_per_rank)
    env["OMP_PROC_BIND"] = "true"
    env["OMP_PLACES"] = "threads"

    cases = " ".join(f"{idx}) CPUS='{c}' ;; " for idx, c in enumerate(rank_cpus))
    wrapper = (
        f'case "$OMPI_COMM_WORLD_RANK" in {cases} *) exit 1;; esac; '
        f'exec taskset -c "$CPUS" {BIN} -v {versao} -b {bloco} -a {arq_a} -B {arq_b} -m'
    )
    cmd = ["mpirun", "--oversubscribe", "-np", str(p), "--bind-to", "none", "bash", "-c", wrapper]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    saida = proc.stdout + "\n" + proc.stderr
    score = int(RE_SCORE.search(saida).group(1))
    leitura, aloc, init, calc, lib, total = map(float, RE_TEMPOS.search(saida).groups())
    return score, leitura, aloc, init, calc, lib, total


def salvar_csv(caminho: str, linhas: list[dict]):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)


def main() -> None:
    os.makedirs(RES_DIR, exist_ok=True)

    # Parte 1: Referencia N=10000 (inclui omp_diag)
    print("=== PARTE 1: Referencia N=10000 (3 reps) ===", flush=True)
    ref_linhas = []
    configs_10k = [
        ("seq", "1", "0", 1, 1, None, None),
        ("omp_diag", "4-fis", "0,2,4,5", 1, 4, None, None),
        ("omp_diag", "8-hib", "0,2,4,5,6,7,8,9", 1, 8, None, None),
        ("omp_diag", "10-fis", "0,2,4,5,6,7,8,9,10,11", 1, 10, None, None),
        ("omp_blocos", "10-fis", "0,2,4,5,6,7,8,9,10,11", 1, 10, 256, None),
        ("omp_tarefas", "10-fis", "0,2,4,5,6,7,8,9,10,11", 1, 10, 256, None),
        ("mpi_blocos", "10-fis", "0,2,4,5,6,7,8,9,10,11", 10, 1, 256, [str(c) for c in [0, 2, 4, 5, 6, 7, 8, 9, 10, 11]]),
        ("hibrido", "5x2-fis", "0,2,4,5,6,7,8,9,10,11", 5, 2, 256, ["0,2", "4,5", "6,7", "8,9", "10,11"]),
    ]

    scores_10k = set()
    for versao, topologia, cpus, p, t, bloco, rank_cpus in configs_10k:
        calcs, tots = [], []
        for rep in range(1, 4):
            if rank_cpus is None:
                score, leit, aloc, init, calc, lib, tot = rodar_omp(versao, 10000, cpus, t, bloco, rep)
            else:
                score, leit, aloc, init, calc, lib, tot = rodar_mpi(versao, 10000, rank_cpus, t, bloco, rep)
            scores_10k.add(score)
            calcs.append(calc)
            tots.append(tot)
            ref_linhas.append({
                "N": 10000, "versao": versao, "topologia": topologia, "cpus": cpus,
                "P": p, "T": t, "bloco": bloco or 0, "rep": rep, "score": score,
                "leitura_s": leit, "alocacao_s": aloc, "inicializacao_s": init,
                "calculo_s": calc, "liberacao_s": lib, "total_s": tot,
            })
        print(
            f"  N=10000 {versao:11s} ({topologia:7s}): calc_med={statistics.mean(calcs):.6f}s "
            f"tot_med={statistics.mean(tots):.6f}s score={sorted(scores_10k)}",
            flush=True,
        )
    assert len(scores_10k) == 1, f"Divergencia N=10000: {scores_10k}"
    salvar_csv(f"{RES_DIR}/exploracao_n10000_ref.csv", ref_linhas)

    # Parte 2: Busca de -b em N=50000 (10-fis: 0,2,4..11)
    print("\n=== PARTE 2: Busca de -b in {128,256,512,1024} em N=50000 (3 reps) ===", flush=True)
    bloc_linhas = []
    scores_50k = set()
    cpus_10 = "0,2,4,5,6,7,8,9,10,11"
    ranks_10 = ["0", "2", "4", "5", "6", "7", "8", "9", "10", "11"]
    ranks_2x5 = ["0,2,4,5,6", "7,8,9,10,11"]
    ranks_5x2 = ["0,2", "4,5", "6,7", "8,9", "10,11"]

    for b in (128, 256, 512, 1024):
        configs_b = [
            ("omp_blocos", "10-fis", cpus_10, 1, 10, None),
            ("omp_tarefas", "10-fis", cpus_10, 1, 10, None),
            ("mpi_blocos", "10-fis", cpus_10, 10, 1, ranks_10),
            ("hibrido", "2x5-fis", cpus_10, 2, 5, ranks_2x5),
            ("hibrido", "5x2-fis", cpus_10, 5, 2, ranks_5x2),
        ]
        for versao, topologia, cpus, p, t, rank_cpus in configs_b:
            calcs, tots = [], []
            for rep in range(1, 4):
                if rank_cpus is None:
                    score, leit, aloc, init, calc, lib, tot = rodar_omp(versao, 50000, cpus, t, b, rep)
                else:
                    score, leit, aloc, init, calc, lib, tot = rodar_mpi(versao, 50000, rank_cpus, t, b, rep)
                scores_50k.add(score)
                calcs.append(calc)
                tots.append(tot)
                bloc_linhas.append({
                    "N": 50000, "versao": versao, "topologia": topologia, "cpus": cpus,
                    "P": p, "T": t, "bloco": b, "rep": rep, "score": score,
                    "leitura_s": leit, "alocacao_s": aloc, "inicializacao_s": init,
                    "calculo_s": calc, "liberacao_s": lib, "total_s": tot,
                })
            print(
                f"  N=50000 {versao:11s} ({topologia:7s}) b={b:4d}: "
                f"calc_med={statistics.mean(calcs):.6f}s tot_med={statistics.mean(tots):.6f}s",
                flush=True,
            )
    assert scores_50k == {32709}, f"Divergencia N=50000: {scores_50k}"
    salvar_csv(f"{RES_DIR}/exploracao_n50000_blocos.csv", bloc_linhas)
    print("Parte 1 e Parte 2 concluidas.", flush=True)


if __name__ == "__main__":
    main()
