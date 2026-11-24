#!/usr/bin/env python3
import csv
import os
import re
import statistics
import subprocess

BIN = "/home/felps/Documents/paralela/trabalho_lcs/lcs_paralelo"
ENTRADAS = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas"
RES_DIR = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/resultados"
OUT_CSV = f"{RES_DIR}/exploracao_n50000_nucleos.csv"
BLOCO = 1024
N = 50000

RE_SCORE = re.compile(r"Score:\s*(\d+)")
RE_TEMPOS = re.compile(
    r"TEMPOS\s+leitura=([\d.]+)\s+alocacao=([\d.]+)\s+inicializacao=([\d.]+)\s+calculo=([\d.]+)\s+liberacao=([\d.]+)\s+total=([\d.]+)"
)

TOPOLOGIAS = [
    ("1", "0", ["0"]),
    ("2", "0,2", ["0", "2"]),
    ("4-SMT", "0,1,2,3", ["0", "1", "2", "3"]),
    ("4-fis", "0,2,4,5", ["0", "2", "4", "5"]),
    ("6", "0,2,4,5,6,7", ["0", "2", "4", "5", "6", "7"]),
    ("8-hib", "0,2,4,5,6,7,8,9", ["0", "2", "4", "5", "6", "7", "8", "9"]),
    ("8-E", "4,5,6,7,8,9,10,11", ["4", "5", "6", "7", "8", "9", "10", "11"]),
    ("10-fis", "0,2,4,5,6,7,8,9,10,11", ["0", "2", "4", "5", "6", "7", "8", "9", "10", "11"]),
    ("12-all", "0,1,2,3,4,5,6,7,8,9,10,11", [str(i) for i in range(12)]),
]

HIBRIDO_CONFIGS = [
    ("1x10-fis", "0,2,4,5,6,7,8,9,10,11", 1, 10, ["0,2,4,5,6,7,8,9,10,11"]),
    ("2x5-fis", "0,2,4,5,6,7,8,9,10,11", 2, 5, ["0,2,4,5,6", "7,8,9,10,11"]),
    ("5x2-fis", "0,2,4,5,6,7,8,9,10,11", 5, 2, ["0,2", "4,5", "6,7", "8,9", "10,11"]),
    ("10x1-fis", "0,2,4,5,6,7,8,9,10,11", 10, 1, ["0", "2", "4", "5", "6", "7", "8", "9", "10", "11"]),
    ("2x4-hib", "0,2,4,5,6,7,8,9", 2, 4, ["0,2,4,5", "6,7,8,9"]),
    ("4x2-hib", "0,2,4,5,6,7,8,9", 4, 2, ["0,2", "4,5", "6,7", "8,9"]),
    ("1x12-all", "0..11", 1, 12, ["0,1,2,3,4,5,6,7,8,9,10,11"]),
    ("2x6-all", "0..11", 2, 6, ["0,1,2,3,4,5", "6,7,8,9,10,11"]),
    ("4x3-all", "0..11", 4, 3, ["0,1,2", "3,4,5", "6,7,8", "9,10,11"]),
    ("12x1-all", "0..11", 12, 1, [str(i) for i in range(12)]),
]


def rodar_omp(versao: str, cpus: str, threads: int, bloco: int | None):
    arq_a = f"{ENTRADAS}/dna_{N}_A.in"
    arq_b = f"{ENTRADAS}/dna_{N}_B.in"
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


def rodar_mpi(versao: str, rank_cpus: list[str], threads_per_rank: int, bloco: int):
    arq_a = f"{ENTRADAS}/dna_{N}_A.in"
    arq_b = f"{ENTRADAS}/dna_{N}_B.in"
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


def main() -> None:
    linhas = []

    def registrar(versao, topologia, cpus, p, t, bloco, rep, vals):
        score, leit, aloc, init, calc, lib, tot = vals
        assert score == 32709, f"Score inesperado: {score}"
        linhas.append({
            "N": N, "versao": versao, "topologia": topologia, "cpus": cpus,
            "P": p, "T": t, "workers": p * t, "bloco": bloco or 0, "rep": rep,
            "score": score, "leitura_s": leit, "alocacao_s": aloc,
            "inicializacao_s": init, "calculo_s": calc, "liberacao_s": lib, "total_s": tot,
        })
        return calc, tot

    print("=== SEQ baseline (N=50000, CPU 0, 3 reps) ===", flush=True)
    seq_c, seq_t = [], []
    for rep in range(1, 4):
        c, t = registrar("seq", "1", "0", 1, 1, None, rep, rodar_omp("seq", "0", 1, None))
        seq_c.append(c)
        seq_t.append(t)
    seq_c_med = statistics.mean(seq_c)
    seq_t_med = statistics.mean(seq_t)
    print(f"  seq (1): calc_med={seq_c_med:.6f}s tot_med={seq_t_med:.6f}s", flush=True)

    for versao in ("omp_blocos", "omp_tarefas", "mpi_blocos"):
        print(f"\n=== {versao} -b {BLOCO} (N=50000, 9 topologias, 3 reps) ===", flush=True)
        for topologia, cpus, rank_list in TOPOLOGIAS:
            calcs, tots = [], []
            for rep in range(1, 4):
                if versao == "mpi_blocos":
                    vals = rodar_mpi(versao, rank_list, 1, BLOCO)
                    p, t = len(rank_list), 1
                else:
                    vals = rodar_omp(versao, cpus, len(rank_list), BLOCO)
                    p, t = 1, len(rank_list)
                c, tot = registrar(versao, topologia, cpus, p, t, BLOCO, rep, vals)
                calcs.append(c)
                tots.append(tot)
            mc = statistics.mean(calcs)
            mt = statistics.mean(tots)
            print(
                f"  {topologia:7s} ({p}x{t:2d}): calc_med={mc:.6f}s (sp={seq_c_med/mc:.3f}x) | "
                f"tot_med={mt:.6f}s (sp={seq_t_med/mt:.3f}x)",
                flush=True,
            )

    print(f"\n=== hibrido -b {BLOCO} (N=50000, P x T, 3 reps) ===", flush=True)
    for topologia, cpus, p, t, rank_cpus in HIBRIDO_CONFIGS:
        calcs, tots = [], []
        for rep in range(1, 4):
            vals = rodar_mpi("hibrido", rank_cpus, t, BLOCO)
            c, tot = registrar("hibrido", topologia, cpus, p, t, BLOCO, rep, vals)
            calcs.append(c)
            tots.append(tot)
        mc = statistics.mean(calcs)
        mt = statistics.mean(tots)
        print(
            f"  {topologia:9s} ({p:2d}x{t:2d}): calc_med={mc:.6f}s (sp={seq_c_med/mc:.3f}x) | "
            f"tot_med={mt:.6f}s (sp={seq_t_med/mt:.3f}x)",
            flush=True,
        )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)
    print(f"\nSalvo em {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
