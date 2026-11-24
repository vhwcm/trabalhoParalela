#!/usr/bin/env python3
import os
import re
import statistics
import subprocess

BIN = "/home/felps/Documents/paralela/trabalho_lcs/lcs_paralelo"
ARQ_A = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas/dna_6000_A.in"
ARQ_B = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas/dna_6000_B.in"

REF_ANTERIOR = [
    ("seq", 1, 0.067725, 0.068641, 0.077649, 0.078568),
    ("omp_diag (4T)", 4, 0.117660, 0.115890, 0.129054, 0.127246),
    ("omp_diag (8T)", 8, 0.129737, 0.131456, 0.144529, 0.147814),
    ("omp_blocos -b 256 (4T)", 4, 0.020418, 0.018078, 0.032122, 0.028889),
    ("omp_blocos -b 256 (8T)", 8, 0.018514, 0.018261, 0.032389, 0.032688),
    ("omp_tarefas -b 256 (4T)", 4, 0.017648, 0.015542, 0.030830, 0.025819),
    ("omp_tarefas -b 256 (8T)", 8, 0.012313, 0.012967, 0.027472, 0.029497),
    ("mpi_blocos -b 256 (4P)", 4, 0.026474, 0.029396, 0.031530, 0.034183),
    ("mpi_blocos -b 256 (8P)", 8, 0.019444, 0.019494, 0.024837, 0.024921),
]

CONFIGS = [
    ("hibrido 2Px2T -b 256", 2, 2, 256),
    ("hibrido 2Px2T -b 128", 2, 2, 128),
    ("hibrido 2Px4T -b 256", 2, 4, 256),
    ("hibrido 2Px4T -b 128", 2, 4, 128),
    ("hibrido 4Px2T -b 256", 4, 2, 256),
    ("hibrido 4Px2T -b 128", 4, 2, 128),
]

RE_SCORE = re.compile(r"Score:\s*(\d+)")
RE_TEMPOS = re.compile(
    r"TEMPOS\s+leitura=([\d.]+)\s+alocacao=([\d.]+)\s+inicializacao=([\d.]+)\s+calculo=([\d.]+)\s+liberacao=([\d.]+)\s+total=([\d.]+)"
)


def executar(nome: str, np: int, nt: int, bloco: int, repeticoes: int = 3):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(nt)
    resultados = []
    for rep in range(1, repeticoes + 1):
        cmd = [
            "mpirun",
            "--oversubscribe",
            "-np",
            str(np),
            BIN,
            "-v",
            "hibrido",
            "-b",
            str(bloco),
            "-a",
            ARQ_A,
            "-B",
            ARQ_B,
            "-m",
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        saida = proc.stdout + "\n" + proc.stderr
        m_score = RE_SCORE.search(saida)
        m_tempos = RE_TEMPOS.search(saida)
        if not m_score or not m_tempos:
            raise RuntimeError(f"Saida inesperada em {nome} rep {rep}:\n{saida}")
        score = int(m_score.group(1))
        leitura, aloc, init, calc, lib, total = map(float, m_tempos.groups())
        resultados.append({
            "rep": rep,
            "score": score,
            "leitura": leitura,
            "alocacao": aloc,
            "inicializacao": init,
            "calculo": calc,
            "liberacao": lib,
            "total": total,
        })
    return resultados


def main() -> None:
    seq_calc_med = REF_ANTERIOR[0][2]
    seq_calc_mediana = REF_ANTERIOR[0][3]
    seq_tot_med = REF_ANTERIOR[0][4]
    seq_tot_mediana = REF_ANTERIOR[0][5]

    linhas_tabela = list(REF_ANTERIOR)

    for nome, np, nt, bloco in CONFIGS:
        res = executar(nome, np, nt, bloco, repeticoes=3)
        scores = {r["score"] for r in res}
        if scores != {3922}:
            raise RuntimeError(f"Score divergente em {nome}: {scores}")
        calcs = [r["calculo"] for r in res]
        tots = [r["total"] for r in res]
        med_c = statistics.mean(calcs)
        mediana_c = statistics.median(calcs)
        med_t = statistics.mean(tots)
        mediana_t = statistics.median(tots)

        print(f"\n=== {nome} (Score={res[0]['score']}) ===")
        for r in res:
            print(
                f"  rep {r['rep']}: calculo={r['calculo']:.6f}s  total={r['total']:.6f}s "
                f"(leitura={r['leitura']:.6f} aloc={r['alocacao']:.6f} init={r['inicializacao']:.6f} lib={r['liberacao']:.6f})"
            )
        print(
            f"  calculo: media={med_c:.6f}s (sp={seq_calc_med/med_c:.4f}x)  "
            f"mediana={mediana_c:.6f}s (sp={seq_calc_mediana/mediana_c:.4f}x)"
        )
        print(
            f"  total:   media={med_t:.6f}s (sp={seq_tot_med/med_t:.4f}x)  "
            f"mediana={mediana_t:.6f}s (sp={seq_tot_mediana/mediana_t:.4f}x)"
        )
        linhas_tabela.append((nome, np * nt, med_c, mediana_c, med_t, mediana_t))

    print("\n=== MINI-TABELA COMPARATIVA PRELIMINAR (N=6000) ===")
    print(f"{'Versao':28s} | {'W':>2s} | {'Calc Med':>9s} | {'Calc Md':>9s} | {'Sp Calc':>8s} | {'Tot Med':>9s} | {'Tot Md':>9s} | {'Sp Tot':>8s}")
    print("-" * 104)
    for nome, w, mc, mdc, mt, mdt in linhas_tabela:
        print(
            f"{nome:28s} | {w:2d} | {mc:9.6f} | {mdc:9.6f} | {seq_calc_med/mc:7.4f}x | "
            f"{mt:9.6f} | {mdt:9.6f} | {seq_tot_med/mt:7.4f}x"
        )


if __name__ == "__main__":
    main()
