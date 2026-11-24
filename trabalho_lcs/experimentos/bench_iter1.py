#!/usr/bin/env python3
import os
import re
import statistics
import subprocess

BIN = "/home/felps/Documents/paralela/trabalho_lcs/lcs_paralelo"
ARQ_A = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas/dna_6000_A.in"
ARQ_B = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas/dna_6000_B.in"

CONFIGS = [
    ("seq", ["-v", "seq"], {}),
    ("omp_diag_4t", ["-v", "omp_diag"], {"OMP_NUM_THREADS": "4"}),
    ("omp_diag_8t", ["-v", "omp_diag"], {"OMP_NUM_THREADS": "8"}),
]

RE_SCORE = re.compile(r"Score:\s*(\d+)")
RE_TEMPOS = re.compile(
    r"TEMPOS\s+leitura=([\d.]+)\s+alocacao=([\d.]+)\s+inicializacao=([\d.]+)\s+calculo=([\d.]+)\s+liberacao=([\d.]+)\s+total=([\d.]+)"
)


def executar(nome: str, args: list[str], extra_env: dict[str, str], repeticoes: int = 3):
    env = os.environ.copy()
    env.update(extra_env)
    resultados = []
    for rep in range(1, repeticoes + 1):
        cmd = [BIN, *args, "-a", ARQ_A, "-B", ARQ_B, "-m"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        saida = proc.stdout + "\n" + proc.stderr
        m_score = RE_SCORE.search(saida)
        m_tempos = RE_TEMPOS.search(saida)
        if not m_score or not m_tempos:
            raise RuntimeError(f"Saida inesperada em {nome} rep {rep}:\n{saida}\n{proc.stderr}")
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
    todos = {}
    scores = set()
    for nome, args, extra_env in CONFIGS:
        res = executar(nome, args, extra_env, repeticoes=3)
        todos[nome] = res
        for r in res:
            scores.add(r["score"])

    print(f"Scores observados: {sorted(scores)}")
    if len(scores) != 1:
        raise RuntimeError(f"Divergencia de Score: {scores}")

    media_seq_calc = statistics.mean(r["calculo"] for r in todos["seq"])
    media_seq_tot = statistics.mean(r["total"] for r in todos["seq"])

    for nome, _, _ in CONFIGS:
        res = todos[nome]
        calcs = [r["calculo"] for r in res]
        tots = [r["total"] for r in res]
        med_c = statistics.mean(calcs)
        mediana_c = statistics.median(calcs)
        med_t = statistics.mean(tots)
        mediana_t = statistics.median(tots)
        sp_c = media_seq_calc / med_c
        sp_t = media_seq_tot / med_t
        print(f"\n=== {nome} (Score={res[0]['score']}) ===")
        for r in res:
            print(
                f"  rep {r['rep']}: calculo={r['calculo']:.6f}s  total={r['total']:.6f}s "
                f"(leitura={r['leitura']:.6f} aloc={r['alocacao']:.6f} init={r['inicializacao']:.6f} lib={r['liberacao']:.6f})"
            )
        print(
            f"  Resumo calculo: media={med_c:.6f}s  mediana={mediana_c:.6f}s  speedup_media={sp_c:.4f}x"
        )
        print(
            f"  Resumo total:   media={med_t:.6f}s  mediana={mediana_t:.6f}s  speedup_media={sp_t:.4f}x"
        )


if __name__ == "__main__":
    main()
