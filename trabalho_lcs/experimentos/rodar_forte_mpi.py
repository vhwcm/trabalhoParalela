#!/usr/bin/env python3
# Escalabilidade forte da versao mpi_blocos (1 thread por processo), 20 execucoes por ponto.
# Reaproveita as funcoes de rodar_experimentos.py e usa a mesma ordem de nucleos da versao OpenMP.

import csv

from rodar_experimentos import BLOCO, RES_DIR, CsvIncremental, executar_serie, stats

TAMANHOS = [15000, 30000, 50000]
PROCESSOS = [
    ("1", ["0"]),
    ("2", ["0", "2"]),
    ("4-fis", ["0", "2", "4", "5"]),
    ("8-hib", ["0", "2", "4", "5", "6", "7", "8", "9"]),
    ("10-fis", ["0", "2", "4", "5", "6", "7", "8", "9", "10", "11"]),
]

CAMPOS_BRUTO = [
    "N", "versao", "topologia", "cpus", "P", "T", "workers", "places", "bloco", "rep", "score",
    "leitura_s", "alocacao_s", "inicializacao_s", "calculo_s", "liberacao_s", "nao_paralelizavel_s", "total_s",
]


def main() -> None:
    bruto = CsvIncremental(f"{RES_DIR}/escalabilidade_forte_mpi_20x_bruto.csv", CAMPOS_BRUTO)
    linhas_resumo = []
    for n in TAMANHOS:
        t1 = None
        for topologia, cpus in PROCESSOS:
            p = len(cpus)
            amostras = executar_serie(bruto, n, "mpi_blocos", topologia, ",".join(cpus), p, 1, "cores", BLOCO, cpus)
            media, dp, cv = stats([a["total"] for a in amostras])
            if t1 is None:
                t1 = media
            speedup = t1 / media
            linhas_resumo.append({
                "N": n, "processos": p, "tot_media_s": f"{media:.6f}", "tot_dp_s": f"{dp:.6f}",
                "tot_cv_pct": f"{cv:.2f}", "speedup": f"{speedup:.4f}", "eficiencia": f"{speedup / p:.4f}",
            })
            print(f"N={n} P={p} total={media:.3f} +- {dp:.3f} speedup={speedup:.2f}", flush=True)
    bruto.close()

    with open(f"{RES_DIR}/escalabilidade_forte_mpi_resumo.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas_resumo[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas_resumo)


if __name__ == "__main__":
    main()
