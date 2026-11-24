#!/usr/bin/env python3
import csv
import re
import subprocess

BIN = "/home/felps/Documents/paralela/trabalho_lcs/lcs_paralelo"
ENTRADAS = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas"
SAIDA_CSV = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/resultados/calibracao_n.csv"

RE_SCORE = re.compile(r"Score:\s*(\d+)")
RE_TEMPOS = re.compile(
    r"TEMPOS\s+leitura=([\d.]+)\s+alocacao=([\d.]+)\s+inicializacao=([\d.]+)\s+calculo=([\d.]+)\s+liberacao=([\d.]+)\s+total=([\d.]+)"
)
RE_RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def main() -> None:
    linhas = []
    for n in (25000, 40000, 50000):
        arq_a = f"{ENTRADAS}/dna_{n}_A.in"
        arq_b = f"{ENTRADAS}/dna_{n}_B.in"
        cmd = ["/usr/bin/time", "-v", "taskset", "-c", "0", BIN, "-v", "seq", "-a", arq_a, "-B", arq_b, "-m"]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        saida = proc.stdout + "\n" + proc.stderr
        score = int(RE_SCORE.search(saida).group(1))
        leitura, aloc, init, calc, lib, total = map(float, RE_TEMPOS.search(saida).groups())
        rss_kb = int(RE_RSS.search(saida).group(1))
        rss_gib = rss_kb / (1024 * 1024)
        print(
            f"N={n}: Score={score} | calculo={calc:.6f}s | total={total:.6f}s | "
            f"init={init:.6f}s | lib={lib:.6f}s | RSS={rss_kb} KiB ({rss_gib:.3f} GiB)"
        )
        linhas.append({
            "N": n,
            "score": score,
            "leitura_s": leitura,
            "alocacao_s": aloc,
            "inicializacao_s": init,
            "calculo_s": calc,
            "liberacao_s": lib,
            "total_s": total,
            "rss_kib": rss_kb,
            "rss_gib": round(rss_gib, 4),
        })

    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)
    print(f"Salvo em {SAIDA_CSV}")


if __name__ == "__main__":
    main()
