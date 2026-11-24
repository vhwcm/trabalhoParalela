#!/usr/bin/env python3
import os
import re
import statistics
import subprocess

BIN = "/home/felps/Documents/paralela/trabalho_lcs/lcs_paralelo"
ARQ_A = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas/dna_6000_A.in"
ARQ_B = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas/dna_6000_B.in"

REF = {
    "seq": {"calc_media": 0.067725, "calc_mediana": 0.068641, "tot_media": 0.077649, "tot_mediana": 0.078568},
    "omp_diag_4t": {"calc_media": 0.117660, "calc_mediana": 0.115890, "tot_media": 0.129054, "tot_mediana": 0.127246},
    "omp_diag_8t": {"calc_media": 0.129737, "calc_mediana": 0.131456, "tot_media": 0.144529, "tot_mediana": 0.147814},
    "omp_blocos_4t": {"calc_media": 0.020418, "calc_mediana": 0.018078, "tot_media": 0.032122, "tot_mediana": 0.028889},
    "omp_blocos_8t": {"calc_media": 0.018514, "calc_mediana": 0.018261, "tot_media": 0.032389, "tot_mediana": 0.032688},
}

CONFIGS = [
    ("omp_tarefas_b256_4t", ["-v", "omp_tarefas", "-b", "256"], {"OMP_NUM_THREADS": "4"}),
    ("omp_tarefas_b256_8t", ["-v", "omp_tarefas", "-b", "256"], {"OMP_NUM_THREADS": "8"}),
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
    for nome, args, extra_env in CONFIGS:
        res = executar(nome, args, extra_env, repeticoes=3)
        scores = {r["score"] for r in res}
        if scores != {3922}:
            raise RuntimeError(f"Score divergente em {nome}: {scores}")
        calcs = [r["calculo"] for r in res]
        tots = [r["total"] for r in res]
        med_c = statistics.mean(calcs)
        mediana_c = statistics.median(calcs)
        med_t = statistics.mean(tots)
        mediana_t = statistics.median(tots)

        suf = "4t" if "4t" in nome else "8t"
        diag_key = f"omp_diag_{suf}"
        bloc_key = f"omp_blocos_{suf}"
        print(f"\n=== {nome} (Score={res[0]['score']}) ===")
        for r in res:
            print(
                f"  rep {r['rep']}: calculo={r['calculo']:.6f}s  total={r['total']:.6f}s "
                f"(leitura={r['leitura']:.6f} aloc={r['alocacao']:.6f} init={r['inicializacao']:.6f} lib={r['liberacao']:.6f})"
            )
        print(f"  calculo: media={med_c:.6f}s  mediana={mediana_c:.6f}s")
        print(f"    speedup vs seq (media/mediana):        {REF['seq']['calc_media'] / med_c:.4f}x / {REF['seq']['calc_mediana'] / mediana_c:.4f}x")
        print(f"    speedup vs {diag_key} (media/mediana): {REF[diag_key]['calc_media'] / med_c:.4f}x / {REF[diag_key]['calc_mediana'] / mediana_c:.4f}x")
        print(f"    speedup vs {bloc_key} (media/mediana): {REF[bloc_key]['calc_media'] / med_c:.4f}x / {REF[bloc_key]['calc_mediana'] / mediana_c:.4f}x")
        print(f"  total:   media={med_t:.6f}s  mediana={mediana_t:.6f}s")
        print(f"    speedup vs seq (media/mediana):        {REF['seq']['tot_media'] / med_t:.4f}x / {REF['seq']['tot_mediana'] / mediana_t:.4f}x")
        print(f"    speedup vs {diag_key} (media/mediana): {REF[diag_key]['tot_media'] / med_t:.4f}x / {REF[diag_key]['tot_mediana'] / mediana_t:.4f}x")
        print(f"    speedup vs {bloc_key} (media/mediana): {REF[bloc_key]['tot_media'] / med_t:.4f}x / {REF[bloc_key]['tot_mediana'] / mediana_t:.4f}x")


if __name__ == "__main__":
    main()
