#!/usr/bin/env python3
"""Bateria oficial de experimentos (Etapa 4B) — 20 execucoes reais por ponto."""

import csv
import math
import os
import re
import statistics
import subprocess
import time

BIN = "/home/felps/Documents/paralela/trabalho_lcs/lcs_paralelo"
ENTRADAS = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas"
RES_DIR = "/home/felps/Documents/paralela/trabalho_lcs/experimentos/resultados"

N_GRANDE = 50000
REPS = 20
BLOCO = 1024

RE_SCORE = re.compile(r"Score:\s*(\d+)")
RE_TEMPOS = re.compile(
    r"TEMPOS\s+leitura=([\d.]+)\s+alocacao=([\d.]+)\s+inicializacao=([\d.]+)\s+calculo=([\d.]+)\s+liberacao=([\d.]+)\s+total=([\d.]+)"
)

TOPOLOGIAS_9 = [
    ("1", "0", 1, "cores"),
    ("2", "0,2", 2, "cores"),
    ("4-SMT", "0,1,2,3", 4, "threads"),
    ("4-fis", "0,2,4,5", 4, "cores"),
    ("6", "0,2,4,5,6,7", 6, "cores"),
    ("8-hib", "0,2,4,5,6,7,8,9", 8, "cores"),
    ("8-E", "4,5,6,7,8,9,10,11", 8, "cores"),
    ("10-fis", "0,2,4,5,6,7,8,9,10,11", 10, "cores"),
    ("12-all", "0,1,2,3,4,5,6,7,8,9,10,11", 12, "threads"),
]

FRACA_PONTOS = [
    ("1", 14000, "0", 1, "cores"),
    ("2", 19799, "0,2", 2, "cores"),
    ("4-fis", 28000, "0,2,4,5", 4, "cores"),
    ("6", 34293, "0,2,4,5,6,7", 6, "cores"),
    ("8-hib", 39598, "0,2,4,5,6,7,8,9", 8, "cores"),
    ("10-fis", 44272, "0,2,4,5,6,7,8,9,10,11", 10, "cores"),
    ("12-all", 48497, "0,1,2,3,4,5,6,7,8,9,10,11", 12, "threads"),
]

SCORES_ESPERADOS: dict[int, int] = {}


def validar_numero(x: float, nome: str) -> None:
    if math.isnan(x) or math.isinf(x) or x < 0.0:
        raise RuntimeError(f"Valor invalido para {nome}: {x}")


def stats(vals: list[float]) -> tuple[float, float, float]:
    media = statistics.mean(vals)
    dp = statistics.stdev(vals) if len(vals) > 1 else 0.0
    cv = (dp / media * 100.0) if media > 0.0 else 0.0
    return media, dp, cv


def rodar_omp(versao: str, n: int, cpus: str, threads: int, places: str, bloco: int | None):
    arq_a = f"{ENTRADAS}/dna_{n}_A.in"
    arq_b = f"{ENTRADAS}/dna_{n}_B.in"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "true"
    env["OMP_PLACES"] = places
    cmd = ["taskset", "-c", cpus, BIN, "-v", versao]
    if bloco is not None:
        cmd += ["-b", str(bloco)]
    cmd += ["-a", arq_a, "-B", arq_b, "-m"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return parse_saida(proc.stdout + "\n" + proc.stderr, n, f"{versao} N={n}")


def rodar_mpi(versao: str, n: int, rank_cpus: list[str], threads_per_rank: int, places: str, bloco: int):
    arq_a = f"{ENTRADAS}/dna_{n}_A.in"
    arq_b = f"{ENTRADAS}/dna_{n}_B.in"
    p = len(rank_cpus)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads_per_rank)
    env["OMP_PROC_BIND"] = "true"
    env["OMP_PLACES"] = places

    cases = " ".join(f"{idx}) CPUS='{c}' ;; " for idx, c in enumerate(rank_cpus))
    wrapper = (
        f'case "$OMPI_COMM_WORLD_RANK" in {cases} *) exit 1;; esac; '
        f'exec taskset -c "$CPUS" {BIN} -v {versao} -b {bloco} -a {arq_a} -B {arq_b} -m'
    )
    cmd = ["mpirun", "--oversubscribe", "-np", str(p), "--bind-to", "none", "bash", "-c", wrapper]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return parse_saida(proc.stdout + "\n" + proc.stderr, n, f"{versao} N={n}")


def parse_saida(saida: str, n: int, ctx: str):
    m_score = RE_SCORE.search(saida)
    m_tempos = RE_TEMPOS.search(saida)
    if not m_score or not m_tempos:
        raise RuntimeError(f"Saida inesperada em {ctx}:\n{saida}")
    score = int(m_score.group(1))
    if score > 65535:
        raise RuntimeError(f"Overflow de unsigned short detectado em {ctx}: Score={score}")
    if n not in SCORES_ESPERADOS:
        SCORES_ESPERADOS[n] = score
    elif SCORES_ESPERADOS[n] != score:
        raise RuntimeError(f"Divergencia de Score em {ctx}: esperado={SCORES_ESPERADOS[n]}, obtido={score}")
    leit, aloc, init, calc, lib, tot = map(float, m_tempos.groups())
    for nome, val in zip(("leitura", "alocacao", "inicializacao", "calculo", "liberacao", "total"), (leit, aloc, init, calc, lib, tot)):
        validar_numero(val, f"{ctx}.{nome}")
    return score, leit, aloc, init, calc, lib, tot


class CsvIncremental:
    def __init__(self, caminho: str, fieldnames: list[str]):
        self.caminho = caminho
        self.f = open(caminho, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()
        self.f.flush()

    def write_row(self, row: dict) -> None:
        self.writer.writerow(row)

    def flush(self) -> None:
        self.f.flush()

    def close(self) -> None:
        self.f.flush()
        self.f.close()


def executar_serie(
    csv_bruto: CsvIncremental,
    n: int,
    versao: str,
    topologia: str,
    cpus: str,
    p: int,
    t: int,
    places: str,
    bloco: int,
    rank_cpus: list[str] | None,
) -> list[dict]:
    amostras = []
    for rep in range(1, REPS + 1):
        if rank_cpus is None:
            score, leit, aloc, init, calc, lib, tot = rodar_omp(versao, n, cpus, t, places, bloco if bloco > 0 else None)
        else:
            score, leit, aloc, init, calc, lib, tot = rodar_mpi(versao, n, rank_cpus, t, places, bloco)
        row = {
            "N": n,
            "versao": versao,
            "topologia": topologia,
            "cpus": cpus,
            "P": p,
            "T": t,
            "workers": p * t,
            "places": places,
            "bloco": bloco,
            "rep": rep,
            "score": score,
            "leitura_s": f"{leit:.6f}",
            "alocacao_s": f"{aloc:.6f}",
            "inicializacao_s": f"{init:.6f}",
            "calculo_s": f"{calc:.6f}",
            "liberacao_s": f"{lib:.6f}",
            "nao_paralelizavel_s": f"{(tot - calc):.6f}",
            "total_s": f"{tot:.6f}",
        }
        csv_bruto.write_row(row)
        amostras.append({
            "score": score, "leitura": leit, "alocacao": aloc, "inicializacao": init,
            "calculo": calc, "liberacao": lib, "nao_par": tot - calc, "total": tot,
        })
    csv_bruto.flush()
    return amostras


def resumir_comparacao(
    nome_versao: str,
    topologia: str,
    n: int,
    p: int,
    t: int,
    bloco: int,
    amostras: list[dict],
    t_seq_calc: float,
    t_seq_tot: float,
) -> dict:
    assert len(amostras) == REPS
    calcs = [a["calculo"] for a in amostras]
    tots = [a["total"] for a in amostras]
    mc, sc, cvc = stats(calcs)
    mt, st, cvt = stats(tots)
    w = p * t
    sp_c = t_seq_calc / mc
    sp_t = t_seq_tot / mt
    return {
        "N": n,
        "versao": nome_versao,
        "topologia": topologia,
        "P": p,
        "T": t,
        "workers": w,
        "bloco": bloco,
        "n_amostras": len(amostras),
        "score": amostras[0]["score"],
        "calc_media_s": f"{mc:.6f}",
        "calc_dp_s": f"{sc:.6f}",
        "calc_cv_pct": f"{cvc:.2f}",
        "tot_media_s": f"{mt:.6f}",
        "tot_dp_s": f"{st:.6f}",
        "tot_cv_pct": f"{cvt:.2f}",
        "speedup_calc": f"{sp_c:.4f}",
        "speedup_tot": f"{sp_t:.4f}",
        "eficiencia_calc": f"{(sp_c / w):.4f}",
        "eficiencia_tot": f"{(sp_t / w):.4f}",
    }


BRUTO_FIELDS = [
    "N", "versao", "topologia", "cpus", "P", "T", "workers", "places", "bloco",
    "rep", "score", "leitura_s", "alocacao_s", "inicializacao_s",
    "calculo_s", "liberacao_s", "nao_paralelizavel_s", "total_s",
]

RESUMO_FIELDS = [
    "N", "versao", "topologia", "P", "T", "workers", "bloco", "n_amostras", "score",
    "calc_media_s", "calc_dp_s", "calc_cv_pct",
    "tot_media_s", "tot_dp_s", "tot_cv_pct",
    "speedup_calc", "speedup_tot", "eficiencia_calc", "eficiencia_tot",
]


def main() -> None:
    t_inicio = time.time()
    os.makedirs(RES_DIR, exist_ok=True)

    # =========================================================================
    # FASE 1: Lei de Amdahl (20x em N_grande = 50000, seq fixado na CPU 0)
    # =========================================================================
    print(f"=== FASE 1: Lei de Amdahl (N={N_GRANDE}, 20x seq CPU 0) ===", flush=True)
    amdahl_bruto = CsvIncremental(f"{RES_DIR}/amdahl_20x_bruto.csv", BRUTO_FIELDS)
    seq_50k = executar_serie(amdahl_bruto, N_GRANDE, "seq", "1", "0", 1, 1, "cores", 0, None)
    amdahl_bruto.close()

    fases = [
        ("leitura_s", "leitura"),
        ("alocacao_s", "alocacao"),
        ("inicializacao_s", "inicializacao"),
        ("calculo_s", "calculo"),
        ("liberacao_s", "liberacao"),
        ("nao_paralelizavel_s", "nao_par"),
        ("total_s", "total"),
    ]
    medias_f1 = {}
    with open(f"{RES_DIR}/amdahl_resumo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "item", "media_s", "dp_s", "cv_pct", "fracao_pct", "speedup_max_amdahl"])
        tot_med = statistics.mean(a["total"] for a in seq_50k)
        for col, key in fases:
            vals = [a[key] for a in seq_50k]
            m, s, cv = stats(vals)
            medias_f1[key] = m
            w.writerow(["fase", col, f"{m:.6f}", f"{s:.6f}", f"{cv:.2f}", f"{(m / tot_med * 100.0):.4f}", ""])
        f_seq = medias_f1["nao_par"] / medias_f1["total"]
        f_par = 1.0 - f_seq
        w.writerow(["fracao", "f_seq (nao_paralelizavel)", f"{medias_f1['nao_par']:.6f}", "", "", f"{(f_seq * 100.0):.4f}", ""])
        w.writerow(["fracao", "f_par (calculo)", f"{medias_f1['calculo']:.6f}", "", "", f"{(f_par * 100.0):.4f}", ""])
        for p_amd in (2, 4, 6, 8, 10, 12):
            sp_max = 1.0 / (f_seq + f_par / p_amd)
            w.writerow(["amdahl_teorico", f"p={p_amd}", "", "", "", "", f"{sp_max:.4f}"])
        w.writerow(["amdahl_teorico", "p=inf", "", "", "", "", f"{(1.0 / f_seq):.4f}"])

    t_seq_50k_calc = medias_f1["calculo"]
    t_seq_50k_tot = medias_f1["total"]
    print(
        f"  seq N={N_GRANDE}: calc={t_seq_50k_calc:.6f}s, tot={t_seq_50k_tot:.6f}s, "
        f"f_seq={(medias_f1['nao_par']/t_seq_50k_tot)*100:.2f}%, S_inf={1.0/(medias_f1['nao_par']/t_seq_50k_tot):.2f}x",
        flush=True,
    )

    # =========================================================================
    # FASE 2A: Comparacao das 6 Estrategias em N = 10000 (20x cada)
    # =========================================================================
    print("\n=== FASE 2A: Comparacao das 6 Estrategias em N=10000 (20x) ===", flush=True)
    comp10_bruto = CsvIncremental(f"{RES_DIR}/comparacao_10k_20x_bruto.csv", BRUTO_FIELDS)
    comp10_resumo = CsvIncremental(f"{RES_DIR}/comparacao_10k_resumo.csv", RESUMO_FIELDS)

    configs_10k = [
        ("seq", "1", "0", 1, 1, "cores", 0, None),
        ("omp_diag", "10-fis", "0,2,4,5,6,7,8,9,10,11", 1, 10, "cores", 0, None),
        ("omp_blocos", "10-fis", "0,2,4,5,6,7,8,9,10,11", 1, 10, "cores", BLOCO, None),
        ("omp_blocos", "12-all", "0,1,2,3,4,5,6,7,8,9,10,11", 1, 12, "threads", BLOCO, None),
        ("omp_tarefas", "10-fis", "0,2,4,5,6,7,8,9,10,11", 1, 10, "cores", BLOCO, None),
        ("omp_tarefas", "12-all", "0,1,2,3,4,5,6,7,8,9,10,11", 1, 12, "threads", BLOCO, None),
        ("mpi_blocos", "8-hib", "0,2,4,5,6,7,8,9", 8, 1, "cores", BLOCO, ["0", "2", "4", "5", "6", "7", "8", "9"]),
        ("mpi_blocos", "10-fis", "0,2,4,5,6,7,8,9,10,11", 10, 1, "cores", BLOCO, ["0", "2", "4", "5", "6", "7", "8", "9", "10", "11"]),
        ("hibrido", "5x2-fis", "0,2,4,5,6,7,8,9,10,11", 5, 2, "cores", BLOCO, ["0,2", "4,5", "6,7", "8,9", "10,11"]),
        ("hibrido", "4x3-all", "0,1,2,3,4,5,6,7,8,9,10,11", 4, 3, "threads", BLOCO, ["0,1,2", "3,4,5", "6,7,8", "9,10,11"]),
    ]
    t_seq_10k_calc = t_seq_10k_tot = 0.0
    for versao, top, cpus, p, t, places, bloco, rcpus in configs_10k:
        am = executar_serie(comp10_bruto, 10000, versao, top, cpus, p, t, places, bloco, rcpus)
        if versao == "seq":
            t_seq_10k_calc = statistics.mean(a["calculo"] for a in am)
            t_seq_10k_tot = statistics.mean(a["total"] for a in am)
        r = resumir_comparacao(versao, top, 10000, p, t, bloco, am, t_seq_10k_calc, t_seq_10k_tot)
        comp10_resumo.write_row(r)
        comp10_resumo.flush()
        print(f"  10k {versao:11s} {top:7s}: calc={r['calc_media_s']}s tot={r['tot_media_s']}s (sp_tot={r['speedup_tot']}x)", flush=True)
    comp10_bruto.close()
    comp10_resumo.close()

    # =========================================================================
    # FASE 2B: Comparacao em N_grande = 50000 (20x cada)
    # =========================================================================
    print(f"\n=== FASE 2B: Comparacao em N={N_GRANDE} (20x) ===", flush=True)
    compg_bruto = CsvIncremental(f"{RES_DIR}/comparacao_grande_20x_bruto.csv", BRUTO_FIELDS)
    configs_grande = [
        ("omp_blocos", "10-fis", "0,2,4,5,6,7,8,9,10,11", 1, 10, "cores", BLOCO, None),
        ("omp_blocos", "12-all", "0,1,2,3,4,5,6,7,8,9,10,11", 1, 12, "threads", BLOCO, None),
        ("mpi_blocos", "8-hib", "0,2,4,5,6,7,8,9", 8, 1, "cores", BLOCO, ["0", "2", "4", "5", "6", "7", "8", "9"]),
        ("mpi_blocos", "10-fis", "0,2,4,5,6,7,8,9,10,11", 10, 1, "cores", BLOCO, ["0", "2", "4", "5", "6", "7", "8", "9", "10", "11"]),
        ("hibrido", "5x2-fis", "0,2,4,5,6,7,8,9,10,11", 5, 2, "cores", BLOCO, ["0,2", "4,5", "6,7", "8,9", "10,11"]),
        ("hibrido", "2x5-fis", "0,2,4,5,6,7,8,9,10,11", 2, 5, "cores", BLOCO, ["0,2,4,5,6", "7,8,9,10,11"]),
        ("hibrido", "4x3-all", "0,1,2,3,4,5,6,7,8,9,10,11", 4, 3, "threads", BLOCO, ["0,1,2", "3,4,5", "6,7,8", "9,10,11"]),
        ("hibrido", "2x6-all", "0,1,2,3,4,5,6,7,8,9,10,11", 2, 6, "threads", BLOCO, ["0,1,2,3,4,5", "6,7,8,9,10,11"]),
    ]
    res_grande_rows = [
        resumir_comparacao("seq", "1", N_GRANDE, 1, 1, 0, seq_50k, t_seq_50k_calc, t_seq_50k_tot)
    ]
    for versao, top, cpus, p, t, places, bloco, rcpus in configs_grande:
        am = executar_serie(compg_bruto, N_GRANDE, versao, top, cpus, p, t, places, bloco, rcpus)
        r = resumir_comparacao(versao, top, N_GRANDE, p, t, bloco, am, t_seq_50k_calc, t_seq_50k_tot)
        res_grande_rows.append(r)
        print(f"  50k {versao:11s} {top:7s}: calc={r['calc_media_s']}s tot={r['tot_media_s']}s (sp_tot={r['speedup_tot']}x)", flush=True)
    compg_bruto.close()

    # =========================================================================
    # FASE 3: Escalabilidade Forte (20x por ponto, omp_tarefas -b 1024)
    # =========================================================================
    print("\n=== FASE 3: Escalabilidade Forte (15k, 30k, 50k; 20x) ===", flush=True)
    forte_bruto = CsvIncremental(f"{RES_DIR}/escalabilidade_forte_20x_bruto.csv", BRUTO_FIELDS)
    forte_resumo = CsvIncremental(f"{RES_DIR}/escalabilidade_forte_resumo.csv", RESUMO_FIELDS)

    for n in (15000, 30000, N_GRANDE):
        if n == N_GRANDE:
            am_seq = seq_50k
            t_seq_c, t_seq_t = t_seq_50k_calc, t_seq_50k_tot
        else:
            am_seq = executar_serie(forte_bruto, n, "seq", "1", "0", 1, 1, "cores", 0, None)
            t_seq_c = statistics.mean(a["calculo"] for a in am_seq)
            t_seq_t = statistics.mean(a["total"] for a in am_seq)
        r_seq = resumir_comparacao("seq", "1", n, 1, 1, 0, am_seq, t_seq_c, t_seq_t)
        forte_resumo.write_row(r_seq)
        forte_resumo.flush()
        print(f"  Forte N={n} seq (1): calc={r_seq['calc_media_s']}s tot={r_seq['tot_media_s']}s", flush=True)

        for top, cpus, workers, places in TOPOLOGIAS_9:
            am = executar_serie(forte_bruto, n, "omp_tarefas", top, cpus, 1, workers, places, BLOCO, None)
            r = resumir_comparacao("omp_tarefas", top, n, 1, workers, BLOCO, am, t_seq_c, t_seq_t)
            forte_resumo.write_row(r)
            forte_resumo.flush()
            print(
                f"  Forte N={n} omp_tarefas {top:6s}: calc={r['calc_media_s']}s tot={r['tot_media_s']}s "
                f"sp_tot={r['speedup_tot']}x eff_tot={r['eficiencia_tot']}",
                flush=True,
            )
            if n == N_GRANDE and top in ("10-fis", "12-all"):
                res_grande_rows.append(r)

    forte_bruto.close()
    forte_resumo.close()

    # Finaliza comparacao_grande_resumo.csv incluindo seq (Fase 1) e omp_tarefas 10-fis/12-all (Fase 3)
    compg_resumo = CsvIncremental(f"{RES_DIR}/comparacao_grande_resumo.csv", RESUMO_FIELDS)
    for r in res_grande_rows:
        compg_resumo.write_row(r)
    compg_resumo.close()

    # =========================================================================
    # FASE 4: Escalabilidade Fraca (20x por ponto, omp_tarefas -b 1024)
    # =========================================================================
    print("\n=== FASE 4: Escalabilidade Fraca (omp_tarefas -b 1024, 20x) ===", flush=True)
    fraca_bruto = CsvIncremental(f"{RES_DIR}/escalabilidade_fraca_20x_bruto.csv", BRUTO_FIELDS)
    fraca_resumo = CsvIncremental(
        f"{RES_DIR}/escalabilidade_fraca_resumo.csv",
        [
            "topologia", "N", "workers", "bloco", "n_amostras", "score",
            "calc_media_s", "calc_dp_s", "calc_cv_pct",
            "tot_media_s", "tot_dp_s", "tot_cv_pct",
            "eficiencia_fraca_calc", "eficiencia_fraca_tot",
        ],
    )

    t1_calc = t1_tot = 0.0
    for top, n, cpus, workers, places in FRACA_PONTOS:
        am = executar_serie(fraca_bruto, n, "omp_tarefas", top, cpus, 1, workers, places, BLOCO, None)
        calcs = [a["calculo"] for a in am]
        tots = [a["total"] for a in am]
        mc, sc, cvc = stats(calcs)
        mt, st, cvt = stats(tots)
        if workers == 1:
            t1_calc, t1_tot = mc, mt
        ef_c = t1_calc / mc
        ef_t = t1_tot / mt
        fraca_resumo.write_row({
            "topologia": top, "N": n, "workers": workers, "bloco": BLOCO,
            "n_amostras": len(am), "score": am[0]["score"],
            "calc_media_s": f"{mc:.6f}", "calc_dp_s": f"{sc:.6f}", "calc_cv_pct": f"{cvc:.2f}",
            "tot_media_s": f"{mt:.6f}", "tot_dp_s": f"{st:.6f}", "tot_cv_pct": f"{cvt:.2f}",
            "eficiencia_fraca_calc": f"{ef_c:.4f}", "eficiencia_fraca_tot": f"{ef_t:.4f}",
        })
        fraca_resumo.flush()
        print(
            f"  Fraca {top:6s} (N={n}): calc={mc:.6f}s (ef={ef_c:.4f}) tot={mt:.6f}s (ef={ef_t:.4f})",
            flush=True,
        )

    fraca_bruto.close()
    fraca_resumo.close()

    duracao = time.time() - t_inicio
    print(f"\nBateria completa finalizada em {duracao:.1f}s. Scores observados: {sorted(SCORES_ESPERADOS.items())}", flush=True)


if __name__ == "__main__":
    main()
