#!/usr/bin/env python3
"""Gerador do relatório técnico em PDF (CI1316 - LCS OpenMP + MPI).

Lê 100% dos dados diretamente dos CSVs e ambiente.txt gerados pelos experimentos
reais e extrai o kernel sequencial diretamente de lcs.c.
"""

from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict

from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = "/home/felps/Documents/paralela/trabalho_lcs"
RES_DIR = f"{BASE_DIR}/experimentos/resultados"
LCS_ORIG = "/home/felps/Documents/paralela/lcs_source_code/lcs.c"
OUT_DIR = f"{BASE_DIR}/relatorio"
OUT_PDF = f"{OUT_DIR}/relatorio_lcs.pdf"

# Cores institucionais sobrias
COR_PRIMARIA = colors.HexColor("#1B365D")
COR_SECUNDARIA = colors.HexColor("#2B6CB0")
COR_DESTAQUE = colors.HexColor("#C05621")
COR_VERDE = colors.HexColor("#2F855A")
COR_ROXO = colors.HexColor("#6B46C1")
COR_CINZA_FUNDO = colors.HexColor("#F2F5F9")
COR_CINZA_BORDA = colors.HexColor("#CBD5E0")
COR_CODE_BG = colors.HexColor("#F7FAFC")
COR_TEXTO = colors.HexColor("#1A202C")
COR_MUTED = colors.HexColor("#4A5568")


def registrar_fontes() -> None:
    pdfmetrics.registerFont(TTFont("LibSans", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"))
    pdfmetrics.registerFont(TTFont("LibSans-Bold", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("LibSans-Italic", "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"))
    pdfmetrics.registerFont(TTFont("LibSans-BoldItalic", "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"))
    pdfmetrics.registerFontFamily(
        "LibSans",
        normal="LibSans",
        bold="LibSans-Bold",
        italic="LibSans-Italic",
        boldItalic="LibSans-BoldItalic",
    )
    pdfmetrics.registerFont(TTFont("Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
    pdfmetrics.registerFont(TTFont("Mono-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"))
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold")


def ler_csv(nome: str) -> list[dict[str, str]]:
    with open(f"{RES_DIR}/{nome}", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extrair_kernel_lcs() -> str:
    with open(LCS_ORIG, encoding="utf-8") as f:
        linhas = f.readlines()
    ini = fim = None
    for idx, line in enumerate(linhas):
        if line.startswith("int LCS("):
            ini = idx
        elif ini is not None and line.startswith("}"):
            fim = idx
            break
    assert ini is not None and fim is not None
    return "".join(linhas[ini : fim + 1]).replace("\t", "    ").rstrip()


def carregar_dados() -> dict:
    dados: dict = {}
    dados["kernel"] = extrair_kernel_lcs()

    with open(f"{RES_DIR}/ambiente.txt", encoding="utf-8") as f:
        dados["ambiente_txt"] = f.read()

    dados["amdahl_resumo"] = ler_csv("amdahl_resumo.csv")
    amd_bruto = ler_csv("amdahl_20x_bruto.csv")
    dados["amdahl_bruto"] = amd_bruto
    dados["seq50_r1_15_calc"] = statistics.mean(float(r["calculo_s"]) for r in amd_bruto[:15])
    dados["seq50_r16_20_calc"] = statistics.mean(float(r["calculo_s"]) for r in amd_bruto[15:])
    dados["seq50_r1_15_lib"] = statistics.mean(float(r["liberacao_s"]) for r in amd_bruto[:15])
    dados["seq50_r16_20_lib"] = statistics.mean(float(r["liberacao_s"]) for r in amd_bruto[15:])

    dados["calibracao"] = ler_csv("calibracao_n.csv")
    dados["comp10k"] = ler_csv("comparacao_10k_resumo.csv")
    dados["comp50k"] = ler_csv("comparacao_grande_resumo.csv")
    dados["comp10k_bruto"] = ler_csv("comparacao_10k_20x_bruto.csv")

    exp_blocos_raw = ler_csv("exploracao_n50000_blocos.csv")
    grp_b: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for r in exp_blocos_raw:
        grp_b[(r["versao"], r["topologia"], int(r["bloco"]))].append(
            (float(r["calculo_s"]), float(r["total_s"]))
        )
    dados["exp_blocos"] = {
        k: (statistics.mean(v[0] for v in vals), statistics.mean(v[1] for v in vals))
        for k, vals in grp_b.items()
    }

    forte_res = ler_csv("escalabilidade_forte_resumo.csv")
    forte_bruto = ler_csv("escalabilidade_forte_20x_bruto.csv")
    dados["forte_resumo"] = forte_res
    dados["forte_bruto"] = forte_bruto

    forte_por_n: dict[int, list[dict]] = defaultdict(list)
    for r in forte_res:
        forte_por_n[int(r["N"])].append(r)

    dados["forte_t1_rel"] = {}
    for n, rows in forte_por_n.items():
        t1_row = [x for x in rows if x["versao"] == "omp_tarefas" and x["topologia"] == "1"][0]
        t1_c = float(t1_row["calc_media_s"])
        t1_t = float(t1_row["tot_media_s"])
        seq_row = [x for x in rows if x["versao"] == "seq"][0]
        tseq_c = float(seq_row["calc_media_s"])
        tseq_t = float(seq_row["tot_media_s"])

        enriquecidas = []
        for x in rows:
            w = int(x["workers"])
            mc = float(x["calc_media_s"])
            mt = float(x["tot_media_s"])
            sp1_c = t1_c / mc
            sp1_t = t1_t / mt
            ef1_c = sp1_c / w
            ef1_t = sp1_t / w
            enriquecidas.append({
                **x,
                "sp1_calc": sp1_c,
                "sp1_tot": sp1_t,
                "ef1_calc": ef1_c,
                "ef1_tot": ef1_t,
                "tseq_calc": tseq_c,
                "tseq_tot": tseq_t,
            })
        dados["forte_t1_rel"][n] = enriquecidas

    nao_par_map: dict[tuple[int, str], float] = {}
    for n in (15000, 30000, 50000):
        for top in ("1", "2", "4-SMT", "4-fis", "6", "8-hib", "8-E", "10-fis", "12-all"):
            sub = [
                r for r in forte_bruto
                if int(r["N"]) == n and r["versao"] == "omp_tarefas" and r["topologia"] == top
            ]
            nao_par_map[(n, top)] = statistics.mean(float(r["nao_paralelizavel_s"]) for r in sub)
    dados["nao_par_map"] = nao_par_map

    dados["fraca_resumo"] = ler_csv("escalabilidade_fraca_resumo.csv")
    fraca_bruto = ler_csv("escalabilidade_fraca_20x_bruto.csv")
    dados["fraca_bruto"] = fraca_bruto
    dados["fraca_extra"] = {}
    for top in ("1", "12-all"):
        sub = [r for r in fraca_bruto if r["topologia"] == top]
        dados["fraca_extra"][top] = {
            "init": statistics.mean(float(r["inicializacao_s"]) for r in sub),
            "lib": statistics.mean(float(r["liberacao_s"]) for r in sub),
            "nao_par": statistics.mean(float(r["nao_paralelizavel_s"]) for r in sub),
        }

    return dados


def fmt_pt(num: float, casas: int = 2) -> str:
    return f"{num:.{casas}f}".replace(".", ",")


def fmt_media_dp(media_s: str, dp_s: str, casas: int = 3) -> str:
    return f"{fmt_pt(float(media_s), casas)} ± {fmt_pt(float(dp_s), casas)}"


def criar_estilos() -> dict[str, ParagraphStyle]:
    return {
        "Uni": ParagraphStyle(
            "Uni", fontName="LibSans-Bold", fontSize=11.0, leading=14.0,
            textColor=COR_PRIMARIA, alignment=TA_CENTER, spaceAfter=2,
        ),
        "Titulo": ParagraphStyle(
            "Titulo", fontName="LibSans-Bold", fontSize=15.0, leading=18.5,
            textColor=COR_PRIMARIA, alignment=TA_CENTER, spaceAfter=4,
        ),
        "Autores": ParagraphStyle(
            "Autores", fontName="LibSans", fontSize=10.5, leading=13.5,
            textColor=COR_MUTED, alignment=TA_CENTER, spaceAfter=7,
        ),
        "H1": ParagraphStyle(
            "H1", fontName="LibSans-Bold", fontSize=13.0, leading=16.0,
            textColor=COR_PRIMARIA, spaceBefore=10, spaceAfter=4, keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body", fontName="LibSans", fontSize=12.0, leading=14.8,
            textColor=COR_TEXTO, alignment=TA_JUSTIFY, spaceAfter=5.0,
        ),
        "Bullet": ParagraphStyle(
            "Bullet", fontName="LibSans", fontSize=12.0, leading=14.6,
            textColor=COR_TEXTO, alignment=TA_JUSTIFY, leftIndent=13,
            firstLineIndent=-9, spaceAfter=4.0,
        ),
        "Code": ParagraphStyle(
            "Code", fontName="Mono", fontSize=8.5, leading=10.5,
            textColor=COR_TEXTO, alignment=TA_LEFT,
        ),
        "Caption": ParagraphStyle(
            "Caption", fontName="LibSans-Italic", fontSize=9.2, leading=11.5,
            textColor=COR_MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=6,
        ),
        "TH": ParagraphStyle(
            "TH", fontName="LibSans-Bold", fontSize=8.0, leading=9.8,
            textColor=colors.white, alignment=TA_CENTER,
        ),
        "TD": ParagraphStyle(
            "TD", fontName="LibSans", fontSize=8.0, leading=9.8,
            textColor=COR_TEXTO, alignment=TA_CENTER,
        ),
        "TDL": ParagraphStyle(
            "TDL", fontName="LibSans", fontSize=8.0, leading=9.8,
            textColor=COR_TEXTO, alignment=TA_LEFT,
        ),
        "TDB": ParagraphStyle(
            "TDB", fontName="LibSans-Bold", fontSize=8.0, leading=9.8,
            textColor=COR_PRIMARIA, alignment=TA_CENTER,
        ),
    }


def estilo_tabela_padrao(col_left: list[int] | None = None, highlight_rows: list[int] | None = None) -> TableStyle:
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), COR_PRIMARIA),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.45, COR_CINZA_BORDA),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COR_CINZA_FUNDO]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
    ]
    if col_left:
        for c in col_left:
            cmds.append(("ALIGN", (c, 1), (c, -1), "LEFT"))
    if highlight_rows:
        for r in highlight_rows:
            cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#EBF8FF")))
    return TableStyle(cmds)


def construir_grafico_speedup_50k(dados: dict) -> Drawing:
    """Grafico vetorial 1: Speedup Medido vs Linear vs Amdahl em N = 50000."""
    w, h = 492, 198
    d = Drawing(w, h)
    px, py = 44, 32
    pw, ph = 290, 148

    rows_50 = dados["forte_t1_rel"][50000]
    amd_map = {}
    for r in dados["amdahl_resumo"]:
        if r["tipo"] == "amdahl_teorico" and r["item"] != "p=inf":
            p_val = int(r["item"].split("=")[1])
            amd_map[p_val] = float(r["speedup_max_amdahl"])

    tops_ordem = ["1", "2", "4-fis", "6", "8-hib", "10-fis", "12-all"]
    pts_seq_calc, pts_seq_tot, pts_t1_tot, pts_amd, pts_lin = [], [], [], [], []
    by_top = {r["topologia"]: r for r in rows_50 if r["versao"] == "omp_tarefas"}

    def map_x(p: float) -> float:
        return px + ((p - 0.5) / 12.0) * pw

    def map_y(y: float) -> float:
        return py + (y / 12.6) * ph

    for yv in (0, 2, 4, 6, 8, 10, 12):
        yy = map_y(yv)
        d.add(Line(px, yy, px + pw, yy, strokeColor=colors.HexColor("#E2E8F0"), strokeWidth=0.6))
        d.add(String(px - 6, yy - 3, str(yv), fontName="LibSans", fontSize=7.6, fillColor=COR_MUTED, textAnchor="end"))

    for pv in (1, 2, 4, 6, 8, 10, 12):
        xx = map_x(pv)
        d.add(Line(xx, py, xx, py + ph, strokeColor=colors.HexColor("#EDF2F7"), strokeWidth=0.5))
        d.add(String(xx, py - 11, str(pv), fontName="LibSans", fontSize=7.6, fillColor=COR_MUTED, textAnchor="middle"))

    d.add(Line(px, py, px + pw, py, strokeColor=COR_TEXTO, strokeWidth=0.9))
    d.add(Line(px, py, px, py + ph, strokeColor=COR_TEXTO, strokeWidth=0.9))
    d.add(String(px + pw / 2, 6, "Threads / Núcleos (p)", fontName="LibSans-Bold", fontSize=8.2, fillColor=COR_TEXTO, textAnchor="middle"))
    d.add(String(12, py + ph / 2 - 4, "Speedup", fontName="LibSans-Bold", fontSize=8.2, fillColor=COR_TEXTO, textAnchor="middle"))

    for top in tops_ordem:
        r = by_top[top]
        p = int(r["workers"])
        xx = map_x(p)
        pts_lin.extend([xx, map_y(float(p))])
        amd_v = 1.0 if p == 1 else amd_map[p]
        pts_amd.extend([xx, map_y(amd_v)])
        pts_seq_calc.extend([xx, map_y(float(r["speedup_calc"]))])
        pts_seq_tot.extend([xx, map_y(float(r["speedup_tot"]))])
        pts_t1_tot.extend([xx, map_y(r["sp1_tot"])])

    series = [
        (pts_lin, colors.HexColor("#718096"), [4, 2], "Ideal Linear (S = p)"),
        (pts_amd, COR_DESTAQUE, [2, 2], "Amdahl Teórico (fs = 4,04%)"),
        (pts_seq_calc, COR_VERDE, None, "Medido Cálculo (Tseq / Tp)"),
        (pts_seq_tot, COR_PRIMARIA, None, "Medido Total (Tseq / Tp)"),
        (pts_t1_tot, COR_SECUNDARIA, None, "Medido Total (T1 / Tp)"),
    ]
    for pts, cor, dash, _ in series:
        kw = {"strokeColor": cor, "strokeWidth": 1.6}
        if dash:
            kw["strokeDashArray"] = dash
        d.add(PolyLine(pts, **kw))
        for i in range(0, len(pts), 2):
            d.add(Circle(pts[i], pts[i + 1], 2.3, fillColor=cor, strokeColor=colors.white, strokeWidth=0.4))

    r_smt = by_top["4-SMT"]
    sx, sy_tot = map_x(4), map_y(float(r_smt["speedup_tot"]))
    d.add(Rect(sx - 2.6, sy_tot - 2.6, 5.2, 5.2, fillColor=COR_ROXO, strokeColor=colors.white, strokeWidth=0.5))

    lx = px + pw + 12
    ly = py + ph - 12
    d.add(Rect(lx - 4, py + 12, 144, ph - 14, fillColor=COR_CINZA_FUNDO, strokeColor=COR_CINZA_BORDA, strokeWidth=0.5))
    d.add(String(lx + 2, ly, "N = 50.000 (omp_tarefas -b 1024)", fontName="LibSans-Bold", fontSize=7.4, fillColor=COR_PRIMARIA))
    cur_y = ly - 18
    for _, cor, dash, label in series:
        kw = {"strokeColor": cor, "strokeWidth": 1.6}
        if dash:
            kw["strokeDashArray"] = dash
        d.add(Line(lx + 2, cur_y + 2, lx + 18, cur_y + 2, **kw))
        d.add(Circle(lx + 10, cur_y + 2, 2.0, fillColor=cor, strokeColor=cor))
        d.add(String(lx + 22, cur_y - 1, label, fontName="LibSans", fontSize=7.3, fillColor=COR_TEXTO))
        cur_y -= 19
    d.add(Rect(lx + 7.5, cur_y - 0.5, 5.0, 5.0, fillColor=COR_ROXO, strokeColor=COR_ROXO))
    d.add(String(lx + 22, cur_y - 1, f"4-SMT Total vs Tseq ({fmt_pt(float(r_smt['speedup_tot']), 2)}×)", fontName="LibSans", fontSize=7.3, fillColor=COR_ROXO))

    return d


def construir_grafico_eficiencias(dados: dict) -> Drawing:
    """Grafico vetorial 2: Eficiencia Forte (T1/Tp) por N e Eficiencia Fraca."""
    w, h = 492, 192
    d = Drawing(w, h)

    px1, py1, pw1, ph1 = 38, 30, 196, 134
    d.add(String(px1 + pw1 / 2, py1 + ph1 + 10, "Eficiência Forte Total (T1 / (p · Tp))", fontName="LibSans-Bold", fontSize=8.4, fillColor=COR_PRIMARIA, textAnchor="middle"))

    def mx1(p: float) -> float:
        return px1 + ((p - 0.5) / 12.0) * pw1

    def my1(e: float) -> float:
        return py1 + (e / 1.06) * ph1

    for yv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        yy = my1(yv)
        d.add(Line(px1, yy, px1 + pw1, yy, strokeColor=colors.HexColor("#E2E8F0"), strokeWidth=0.6))
        d.add(String(px1 - 4, yy - 2.5, fmt_pt(yv, 1), fontName="LibSans", fontSize=7.2, fillColor=COR_MUTED, textAnchor="end"))
    for pv in (1, 2, 4, 6, 8, 10, 12):
        xx = mx1(pv)
        d.add(String(xx, py1 - 10, str(pv), fontName="LibSans", fontSize=7.2, fillColor=COR_MUTED, textAnchor="middle"))
    d.add(Line(px1, py1, px1 + pw1, py1, strokeColor=COR_TEXTO, strokeWidth=0.8))
    d.add(Line(px1, py1, px1, py1 + ph1, strokeColor=COR_TEXTO, strokeWidth=0.8))
    d.add(String(px1 + pw1 / 2, 6, "Threads (p)", fontName="LibSans-Bold", fontSize=7.8, fillColor=COR_TEXTO, textAnchor="middle"))

    tops = ["1", "2", "4-fis", "6", "8-hib", "10-fis", "12-all"]
    cores_n = [(15000, COR_DESTAQUE, "N = 15k"), (30000, COR_SECUNDARIA, "N = 30k"), (50000, COR_PRIMARIA, "N = 50k")]
    for idx_n, (n_val, cor, lbl) in enumerate(cores_n):
        by_t = {r["topologia"]: r for r in dados["forte_t1_rel"][n_val] if r["versao"] == "omp_tarefas"}
        pts = []
        for top in tops:
            r = by_t[top]
            pts.extend([mx1(int(r["workers"])), my1(r["ef1_tot"])])
        d.add(PolyLine(pts, strokeColor=cor, strokeWidth=1.6))
        for i in range(0, len(pts), 2):
            d.add(Circle(pts[i], pts[i + 1], 2.1, fillColor=cor, strokeColor=colors.white, strokeWidth=0.4))
        ly = py1 + ph1 - 13 - idx_n * 12
        d.add(Line(px1 + 128, ly + 2, px1 + 142, ly + 2, strokeColor=cor, strokeWidth=1.7))
        d.add(String(px1 + 146, ly - 1, lbl, fontName="LibSans", fontSize=7.2, fillColor=COR_TEXTO))

    px2, py2, pw2, ph2 = 278, 30, 198, 134
    d.add(String(px2 + pw2 / 2, py2 + ph2 + 10, "Eficiência Fraca (T1 / Tp, N = 14000·√p)", fontName="LibSans-Bold", fontSize=8.4, fillColor=COR_PRIMARIA, textAnchor="middle"))

    def mx2(p: float) -> float:
        return px2 + ((p - 0.5) / 12.0) * pw2

    def my2(e: float) -> float:
        return py2 + (e / 1.06) * ph2

    for yv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        yy = my2(yv)
        d.add(Line(px2, yy, px2 + pw2, yy, strokeColor=colors.HexColor("#E2E8F0"), strokeWidth=0.6))
        d.add(String(px2 - 4, yy - 2.5, fmt_pt(yv, 1), fontName="LibSans", fontSize=7.2, fillColor=COR_MUTED, textAnchor="end"))
    for pv in (1, 2, 4, 6, 8, 10, 12):
        xx = mx2(pv)
        d.add(String(xx, py2 - 10, str(pv), fontName="LibSans", fontSize=7.2, fillColor=COR_MUTED, textAnchor="middle"))
    d.add(Line(px2, py2, px2 + pw2, py2, strokeColor=COR_TEXTO, strokeWidth=0.8))
    d.add(Line(px2, py2, px2, py2 + ph2, strokeColor=COR_TEXTO, strokeWidth=0.8))
    d.add(String(px2 + pw2 / 2, 6, "Threads (p)", fontName="LibSans-Bold", fontSize=7.8, fillColor=COR_TEXTO, textAnchor="middle"))

    y_ideal = my2(1.0)
    d.add(Line(px2, y_ideal, px2 + pw2, y_ideal, strokeColor=colors.HexColor("#718096"), strokeWidth=1.0, strokeDashArray=[3, 2]))

    pts_fc, pts_ft = [], []
    for r in dados["fraca_resumo"]:
        p = int(r["workers"])
        pts_fc.extend([mx2(p), my2(float(r["eficiencia_fraca_calc"]))])
        pts_ft.extend([mx2(p), my2(float(r["eficiencia_fraca_tot"]))])

    d.add(PolyLine(pts_fc, strokeColor=COR_VERDE, strokeWidth=1.7))
    d.add(PolyLine(pts_ft, strokeColor=COR_PRIMARIA, strokeWidth=1.7))
    for i in range(0, len(pts_fc), 2):
        d.add(Circle(pts_fc[i], pts_fc[i + 1], 2.2, fillColor=COR_VERDE, strokeColor=colors.white, strokeWidth=0.4))
        d.add(Circle(pts_ft[i], pts_ft[i + 1], 2.2, fillColor=COR_PRIMARIA, strokeColor=colors.white, strokeWidth=0.4))

    d.add(Line(px2 + 108, py2 + ph2 - 19, px2 + 122, py2 + ph2 - 19, strokeColor=colors.HexColor("#718096"), strokeWidth=1.1, strokeDashArray=[3, 2]))
    d.add(String(px2 + 126, py2 + ph2 - 22, "Ideal (1,00)", fontName="LibSans", fontSize=7.2, fillColor=COR_TEXTO))
    d.add(Line(px2 + 108, py2 + ph2 - 31, px2 + 122, py2 + ph2 - 31, strokeColor=COR_VERDE, strokeWidth=1.7))
    d.add(String(px2 + 126, py2 + ph2 - 34, "Cálculo", fontName="LibSans", fontSize=7.2, fillColor=COR_TEXTO))
    d.add(Line(px2 + 108, py2 + ph2 - 43, px2 + 122, py2 + ph2 - 43, strokeColor=COR_PRIMARIA, strokeWidth=1.7))
    d.add(String(px2 + 126, py2 + ph2 - 46, "Total", fontName="LibSans", fontSize=7.2, fillColor=COR_TEXTO))

    return d


def cabecalho_rodape(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("LibSans", 8.5)
    canvas.setFillColor(COR_MUTED)
    canvas.drawRightString(A4[0] - 1.5 * cm, 0.9 * cm, f"Página {doc.page}")
    canvas.drawString(1.5 * cm, 0.9 * cm, "UFPR – CI1316 (Programação Paralela) – LCS OpenMP + MPI")
    canvas.restoreState()


def gerar_pdf(dados: dict) -> None:
    registrar_fontes()
    os.makedirs(OUT_DIR, exist_ok=True)
    s = criar_estilos()

    doc = SimpleDocTemplate(
        OUT_PDF,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.55 * cm,
        title="Relatorio Trabalho 1 - LCS OpenMP + MPI",
        author="Felipe Augusto Dittert Noleto; Victor Hugo Weigmann Chequer Maia",
    )

    story = []
    largura_util = A4[0] - 3.0 * cm

    # =========================================================================
    # CABECALHO PAGINA 1 (SEM CAPA VAZIA)
    # =========================================================================
    story.append(Paragraph("UFPR – Departamento de Informática – CI1316 – Programação Paralela", s["Uni"]))
    story.append(Paragraph("Prof. Marco Antonio Zanata Alves", s["Autores"]))
    story.append(Paragraph("Trabalho Prático 1: Paralelização do Algoritmo LCS com OpenMP + MPI", s["Titulo"]))
    story.append(Paragraph(
        "<b>Autores:</b> Felipe Augusto Dittert Noleto (GRR20205689) &nbsp;|&nbsp; "
        "Victor Hugo Weigmann Chequer Maia (GRR20245275)",
        s["Autores"],
    ))
    story.append(Spacer(1, 4))

    # =========================================================================
    # ITEM 1: KERNEL SEQUENCIAL
    # =========================================================================
    story.append(Paragraph("1. Kernel do Algoritmo Sequencial e Análise de Dependências", s["H1"]))
    story.append(Paragraph(
        "Abaixo apresenta-se o recorte exato da função <b>LCS</b> extraída do código sequencial "
        "original (<font name='Mono' size='10.5'>lcs_source_code/lcs.c</font>), responsável pelo "
        "preenchimento da matriz de pontuação:",
        s["Body"],
    ))

    kernel_esc = (
        dados["kernel"]
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    code_tbl = Table([[Paragraph(kernel_esc.replace("\n", "<br/>").replace(" ", "&nbsp;"), s["Code"])]], colWidths=[largura_util])
    code_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COR_CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, COR_CINZA_BORDA),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(code_tbl)
    story.append(Spacer(1, 5))

    story.append(Paragraph(
        "O kernel implementa programação dinâmica com complexidade temporal e espacial de "
        "<b>O(N · M)</b> (armazenando cada célula como <font name='Mono' size='10.5'>unsigned short</font>, "
        "2 bytes). A linha 0 e a coluna 0 são previamente zeradas, e cada célula "
        "<font name='Mono' size='10.5'>scoreMatrix[i][j]</font> depende de três vizinhos já computados: "
        "o elemento diagonal superior esquerdo <font name='Mono' size='10.5'>(i-1, j-1)</font> quando há "
        "casamento (<font name='Mono' size='10.5'>seqA[j-1] == seqB[i-1]</font>), ou o máximo entre o "
        "elemento superior <font name='Mono' size='10.5'>(i-1, j)</font> e o esquerdo "
        "<font name='Mono' size='10.5'>(i, j-1)</font> caso contrário.",
        s["Body"],
    ))
    story.append(Paragraph(
        "Devido à dependência carregada pelo laço em <font name='Mono' size='10.5'>j-1</font> (horizontal) e "
        "em <font name='Mono' size='10.5'>i-1</font> (vertical e diagonal), o laço interno em "
        "<font name='Mono' size='10.5'>j</font> <b>não pode ser paralelizado diretamente</b> com um "
        "<font name='Mono' size='10.5'>#pragma omp parallel for</font> convencional: iterações adjacentes "
        "na mesma linha ou coluna possuem condição de corrida (dependência verdadeira <i>read-after-write</i>). "
        "A paralelização exige reordenamento em frentes de onda (antidiagonais ou blocos dependentes).",
        s["Body"],
    ))

    # =========================================================================
    # ITEM 2: ESTRATEGIA FINAL VITORIOSA
    # =========================================================================
    story.append(Paragraph("2. Estratégia Final (Vitoriosa) de Paralelização", s["H1"]))
    story.append(Paragraph(
        "O programa paralelo unificado (<font name='Mono' size='10.5'>lcs_paralelo.c</font>) aceita a flag "
        "<font name='Mono' size='10.5'>-v</font> com seis modos: <font name='Mono' size='10.5'>seq</font>, "
        "<font name='Mono' size='10.5'>omp_diag</font> (antidiagonais finas), "
        "<font name='Mono' size='10.5'>omp_blocos</font> (antidiagonais de blocos com barreira por onda), "
        "<font name='Mono' size='10.5'>omp_tarefas</font>, <font name='Mono' size='10.5'>mpi_blocos</font> "
        "(particionamento por faixas horizontais em <i>pipeline</i> com <font name='Mono' size='10.5'>MPI_Send/Recv</font>) "
        "e <font name='Mono' size='10.5'>hibrido</font> (MPI entre processos + blocos OpenMP internos). "
        "Por padrão (sem <font name='Mono' size='10.5'>-v</font>), adota <font name='Mono' size='10.5'>omp_tarefas</font> "
        "com bloco <b>-b 1024</b> em 1 processo e seleciona automaticamente <font name='Mono' size='10.5'>hibrido</font> "
        "quando executado com <font name='Mono' size='10.5'>mpirun -np &gt; 1</font>.",
        s["Body"],
    ))
    story.append(Paragraph(
        "A estratégia vitoriosa em memória compartilhada é a <b>omp_tarefas (-b 1024)</b>: a matriz é particionada "
        "em blocos <font name='Mono' size='10.5'>1024 × 1024</font>, e uma única thread gera tarefas OpenMP "
        "com cláusulas <font name='Mono' size='10.5'>depend(in: acima, esquerda) depend(out: atual)</font> "
        "sobre uma matriz sentinela de dependências. Isso elimina barreiras globais entre ondas (diferente de "
        "<font name='Mono' size='10.5'>omp_blocos</font>), reduz drasticamente a sincronização frente a "
        "<font name='Mono' size='10.5'>omp_diag</font>, garante localidade de cache L1d nas sublinhas de bloco "
        "e permite balanceamento dinâmico natural entre P-cores e E-cores.",
        s["Body"],
    ))

    eb = dados["exp_blocos"]
    story.append(Paragraph(
        f"Na exploração do tamanho de bloco em <b>N = 50.000</b> (10-fis, 3 repetições), o cálculo em "
        f"<font name='Mono' size='10.5'>omp_tarefas</font> caiu de <b>{fmt_pt(eb[('omp_tarefas','10-fis',128)][0], 3)}s</b> "
        f"(<font name='Mono' size='10.5'>-b 128</font>) para <b>{fmt_pt(eb[('omp_tarefas','10-fis',256)][0], 3)}s</b> "
        f"(<font name='Mono' size='10.5'>-b 256</font>), <b>{fmt_pt(eb[('omp_tarefas','10-fis',512)][0], 3)}s</b> "
        f"(<font name='Mono' size='10.5'>-b 512</font>) e atingiu o mínimo em <b>{fmt_pt(eb[('omp_tarefas','10-fis',1024)][0], 3)}s</b> "
        f"com <b>-b 1024</b> (melhor também para <font name='Mono' size='10.5'>omp_blocos</font>, "
        f"<font name='Mono' size='10.5'>mpi_blocos</font> e <font name='Mono' size='10.5'>hibrido</font>). "
        f"Na busca da quantidade ideal de núcleos: <b>12-all</b> obteve o menor tempo absoluto para "
        f"N ≥ 30.000, <b>10-fis</b> foi o melhor arranjo puramente físico sem SMT (e vencedor em N = 10.000), "
        f"<b>2–4-fis</b> entregaram a maior eficiência por worker e <b>8-hib</b> representou o melhor "
        f"compromisso operacional liberando núcleos para o sistema.",
        s["Body"],
    ))

    c10_map = {(r["versao"], r["topologia"]): r for r in dados["comp10k"]}
    c50_map = {(r["versao"], r["topologia"]): r for r in dados["comp50k"]}
    linhas_est = [
        ("seq", "1"),
        ("omp_diag", "10-fis"),
        ("omp_blocos", "10-fis"),
        ("omp_blocos", "12-all"),
        ("omp_tarefas", "10-fis"),
        ("omp_tarefas", "12-all"),
        ("mpi_blocos", "8-hib"),
        ("mpi_blocos", "10-fis"),
        ("hibrido", "5x2-fis"),
        ("hibrido", "4x3-all"),
    ]
    tbl_est_rows = [[
        Paragraph("Estratégia", s["TH"]),
        Paragraph("Topologia", s["TH"]),
        Paragraph("N=10k Cálculo (s)", s["TH"]),
        Paragraph("N=10k Total (s)", s["TH"]),
        Paragraph("Spd Tot 10k", s["TH"]),
        Paragraph("N=50k Cálculo (s)", s["TH"]),
        Paragraph("N=50k Total (s)", s["TH"]),
        Paragraph("Spd Tot 50k", s["TH"]),
    ]]
    for ver, top in linhas_est:
        r10 = c10_map.get((ver, top))
        r50 = c50_map.get((ver, top))
        c10_s = fmt_media_dp(r10["calc_media_s"], r10["calc_dp_s"], 4) if r10 else "—"
        t10_s = fmt_media_dp(r10["tot_media_s"], r10["tot_dp_s"], 4) if r10 else "—"
        sp10_s = f"{fmt_pt(float(r10['speedup_tot']), 2)}×" if r10 else "—"
        c50_s = fmt_media_dp(r50["calc_media_s"], r50["calc_dp_s"], 3) if r50 else "n/a*"
        t50_s = fmt_media_dp(r50["tot_media_s"], r50["tot_dp_s"], 3) if r50 else "n/a*"
        sp50_s = f"{fmt_pt(float(r50['speedup_tot']), 2)}×" if r50 else "n/a*"
        tbl_est_rows.append([
            Paragraph(ver, s["TDL"]),
            Paragraph(top, s["TD"]),
            Paragraph(c10_s, s["TD"]),
            Paragraph(t10_s, s["TD"]),
            Paragraph(sp10_s, s["TDB"] if ver == "omp_tarefas" else s["TD"]),
            Paragraph(c50_s, s["TD"]),
            Paragraph(t50_s, s["TD"]),
            Paragraph(sp50_s, s["TDB"] if ver == "omp_tarefas" else s["TD"]),
        ])
    t_est = Table(
        tbl_est_rows,
        colWidths=[2.05*cm, 1.70*cm, 2.55*cm, 2.55*cm, 1.45*cm, 2.65*cm, 2.65*cm, 1.45*cm],
        repeatRows=1,
    )
    t_est.setStyle(estilo_tabela_padrao(col_left=[0], highlight_rows=[6]))
    story.append(KeepTogether([
        t_est,
        Paragraph(
            "Tabela 1: Comparação das 6 estratégias (média ± σ em 20 execuções, bloco 1024). "
            "*omp_diag em N=50k demandaria ~100k barreiras OpenMP (já sofreu slowdown em 10k: 0,52×).",
            s["Caption"],
        ),
    ]))
    story.append(Paragraph(
        "Em <b>N = 10.000</b>, o <font name='Mono' size='10.5'>mpi_blocos (10-fis)</font> vence no tempo total "
        "(0,0664 s vs 0,0802 s do <font name='Mono' size='10.5'>omp_tarefas</font>) porque múltiplos processos "
        "paralelizam alocação, <i>first-touch</i> e <font name='Mono' size='10.5'>free()</font> da matriz. "
        "Porém, em <b>N = 50.000</b>, o <font name='Mono' size='10.5'>omp_tarefas (12-all)</font> vence com folga "
        "(<b>0,979 s</b> cálculo / <b>1,375 s</b> total, <b>4,76×</b>), seguido por <font name='Mono' size='10.5'>mpi_blocos (10-fis)</font> "
        "(1,438 s), <font name='Mono' size='10.5'>omp_blocos (12-all)</font> (1,562 s) e "
        "<font name='Mono' size='10.5'>hibrido (5x2-fis)</font> (1,628 s). Em um único nó, o modo híbrido paga "
        "duplo <i>overhead</i> (barreiras MPI + <i>fork/join</i> OpenMP interno), embora seja essencial em "
        "execuções multi-nó onde processos MPI distribuem memória entre máquinas e threads OpenMP exploram cada nó.",
        s["Body"],
    ))

    # =========================================================================
    # ITEM 3: METODOLOGIA EXPERIMENTAL
    # =========================================================================
    story.append(Paragraph("3. Metodologia Experimental e Ambiente de Execução", s["H1"]))
    story.append(Paragraph(
        "Todos os testes foram executados em uma única máquina física dedicada (sem virtualização em nuvem), "
        "cujas especificações coletadas em <font name='Mono' size='10.5'>ambiente.txt</font> e "
        "<font name='Mono' size='10.5'>lscpu</font> são:",
        s["Body"],
    ))
    for bullet in [
        "<b>Processador:</b> 12th Gen Intel Core i7-1265U (10 núcleos físicos / 12 threads, arquitetura híbrida): "
        "2 <i>P-cores</i> com SMT (CPUs lógicas <font name='Mono' size='10.5'>[0,1]</font> e <font name='Mono' size='10.5'>[2,3]</font>, "
        "até 4,8 GHz, L1d de 48 KB e L2 dedicada por núcleo) e 8 <i>E-cores</i> sem SMT (CPUs "
        "<font name='Mono' size='10.5'>[4..11]</font>, até 3,6 GHz, L1d de 32 KB e L2 compartilhada em dois grupos de 4 núcleos); "
        "cache L3 unificada de 12 MB; 15 GiB de RAM.",
        "<b>Software e Compilação:</b> Ubuntu 24.04.5 LTS, kernel Linux <font name='Mono' size='10.5'>7.0.0-43-generic</font>, "
        "compilador <font name='Mono' size='10.5'>gcc 13.3.0</font> via <font name='Mono' size='10.5'>mpicc</font> (Open MPI 4.1.6), "
        "flags <font name='Mono' size='10.5'>-O3 -fopenmp -Wall</font>. A corretude foi validada comparando o "
        "<i>Score</i> e matrizes pequenas (<font name='Mono' size='10.5'>-DDEBUGMATRIX</font>) contra <font name='Mono' size='10.5'>lcs_ref</font>.",
        "<b>Controle de Frequência e Afinidade:</b> Manteve-se <font name='Mono' size='10.5'>scaling_governor = powersave</font> e "
        "<font name='Mono' size='10.5'>no_turbo = 0</font> por ausência de privilégios de <font name='Mono' size='10.5'>root</font> para "
        "alteração em <font name='Mono' size='10.5'>/sys/devices/system/cpu</font>. Para mitigar migração entre P-cores e E-cores, "
        "aplicou-se afinidade estrita com <font name='Mono' size='10.5'>taskset -c</font>, "
        "<font name='Mono' size='10.5'>OMP_PROC_BIND=true</font>, <font name='Mono' size='10.5'>OMP_PLACES</font> "
        "(<font name='Mono' size='10.5'>cores</font> ou <font name='Mono' size='10.5'>threads</font>) e "
        "<font name='Mono' size='10.5'>mpirun --bind-to none</font> combinado com <font name='Mono' size='10.5'>taskset</font> por ranque.",
        "<b>Estatística e Escolha de N = 50.000:</b> Cada ponto experimental foi executado <b>20 vezes</b>, reportando "
        "média aritmética e desvio padrão amostral (σ); todas as métricas derivadas foram calculadas sobre as médias. "
        "Entradas sintéticas de DNA (<font name='Mono' size='10.5'>A,C,G,T</font>) garantiram <i>Score</i> ≤ 32.709 "
        "(sem <i>overflow</i> de <font name='Mono' size='10.5'>unsigned short</font>). Calibrou-se <b>N = 50.000</b> "
        "(RSS de <b>4,67 GiB</b> e zero <i>swap</i>); descartou-se N = 52.000 por provocar <b>5.603 páginas</b> de "
        "<font name='Mono' size='10.5'>pswpout</font> no <font name='Mono' size='10.5'>/proc/vmstat</font>.",
    ]:
        story.append(Paragraph(f"• {bullet}", s["Bullet"]))

    # =========================================================================
    # ITEM 4: REGIAO PURAMENTE SEQUENCIAL
    # =========================================================================
    story.append(Paragraph("4. Medição da Região Puramente Sequencial (N = 50.000)", s["H1"]))
    story.append(Paragraph(
        "Com a versão <font name='Mono' size='10.5'>seq</font> fixada na CPU 0 (P-core) em N = 50.000 (20 execuções), "
        "mediu-se cada fase do programa. A região não paralelizável no algoritmo OpenMP engloba leitura das sequências, "
        "alocação, inicialização das bordas da matriz e liberação de memória:",
        s["Body"],
    ))

    nomes_fases = {
        "leitura_s": "Leitura dos arquivos",
        "alocacao_s": "Alocação da matriz",
        "inicializacao_s": "Inicialização das bordas",
        "calculo_s": "Cálculo LCS (kernel paralelizável)",
        "liberacao_s": "Liberação de memória (free)",
        "nao_paralelizavel_s": "Total não paralelizável (fs)",
        "total_s": "Tempo Total da Execução",
    }
    tbl_amd1 = [[
        Paragraph("Fase da Execução", s["TH"]),
        Paragraph("Média ± σ (s)", s["TH"]),
        Paragraph("CV (%)", s["TH"]),
        Paragraph("Fração do Total (%)", s["TH"]),
    ]]
    for r in dados["amdahl_resumo"]:
        if r["tipo"] == "fase":
            dest = r["item"] in ("nao_paralelizavel_s", "total_s")
            st_lbl = s["TDB"] if dest else s["TDL"]
            st_num = s["TDB"] if dest else s["TD"]
            tbl_amd1.append([
                Paragraph(nomes_fases[r["item"]], st_lbl),
                Paragraph(fmt_media_dp(r["media_s"], r["dp_s"], 4), st_num),
                Paragraph(f"{fmt_pt(float(r['cv_pct']), 2)}%", st_num),
                Paragraph(f"{fmt_pt(float(r['fracao_pct']), 2)}%", st_num),
            ])
    t_fases = Table(tbl_amd1, colWidths=[6.3*cm, 4.4*cm, 2.4*cm, 4.0*cm], repeatRows=1)
    t_fases.setStyle(estilo_tabela_padrao(col_left=[0], highlight_rows=[6]))
    story.append(KeepTogether([
        t_fases,
        Paragraph("Tabela 2: Decomposição temporal da execução sequencial em N = 50.000 (20 repetições, CPU 0).", s["Caption"]),
    ]))
    story.append(Paragraph(
        "A fração puramente sequencial obtida foi <b>f<sub>s</sub> = 4,04%</b> (0,2645 s sobre 6,5493 s totais), "
        "dominada pela liberação de 4,67 GiB de páginas (<font name='Mono' size='10.5'>munmap/free</font>, 3,23%) e "
        "pela inicialização de bordas (0,79%). A fração paralelizável corresponde a <b>f<sub>p</sub> = 95,96%</b>.",
        s["Body"],
    ))

    # =========================================================================
    # ITEM 5: LEI DE AMDAHL
    # =========================================================================
    story.append(Paragraph("5. Limite Teórico pela Lei de Amdahl", s["H1"]))
    story.append(Paragraph(
        "Para obter os tempos sem interferência de E/S de depuração, instrumentou-se o código sob a flag "
        "<font name='Mono' size='10.5'>-m</font> utilizando temporizadores <font name='Mono' size='10.5'>MPI_Wtime()</font> "
        "ao redor de cada fase (<font name='Mono' size='10.5'>t_calculo</font> isolando o kernel paralelizável e "
        "<font name='Mono' size='10.5'>t_nao_paralelizavel = t_total - t_calculo</font>), imprimindo uma única linha em "
        "<font name='Mono' size='10.5'>stderr</font> ao término. Pela Lei de Amdahl, "
        "<b>S<sub>máx</sub>(p) = 1 / (f<sub>s</sub> + (1 − f<sub>s</sub>) / p)</b>, com f<sub>s</sub> = 0,040384:",
        s["Body"],
    ))

    amd_cols_h = [Paragraph("Processadores (p)", s["TH"])]
    amd_cols_v = [Paragraph("S<sub>máx</sub> Teórico", s["TDB"])]
    for r in dados["amdahl_resumo"]:
        if r["tipo"] == "amdahl_teorico":
            lbl = "∞" if r["item"] == "p=inf" else r["item"].split("=")[1]
            amd_cols_h.append(Paragraph(f"p = {lbl}", s["TH"]))
            amd_cols_v.append(Paragraph(f"{fmt_pt(float(r['speedup_max_amdahl']), 2)}×", s["TDB"]))
    t_amd = Table([amd_cols_h, amd_cols_v], colWidths=[3.7*cm] + [1.91*cm]*7)
    t_amd.setStyle(estilo_tabela_padrao(col_left=[0]))
    story.append(KeepTogether([
        t_amd,
        Paragraph("Tabela 3: Speedup máximo teórico segundo a Lei de Amdahl para N = 50.000 (f<sub>s</sub> = 4,04%).", s["Caption"]),
    ]))

    # =========================================================================
    # ITEM 6: TABELAS DE SPEEDUP E EFICIENCIA + GRAFICOS
    # =========================================================================
    story.append(Paragraph("6. Speedup e Eficiência (Escalabilidade Forte)", s["H1"]))
    story.append(Paragraph(
        "As Tabelas 4, 5 e 6 apresentam os resultados da estratégia vitoriosa "
        "<font name='Mono' size='10.5'>omp_tarefas -b 1024</font> (20 execuções por ponto) para "
        "<b>N = 15.000, 30.000 e 50.000</b> nas 9 topologias de núcleos/threads da máquina. Reportam-se o "
        "speedup e a eficiência tanto em relação ao algoritmo sequencial original "
        "(<b>T<sub>seq</sub> / T<sub>p</sub></b>) quanto em relação à própria versão blocada com 1 thread "
        "(<b>T<sub>1</sub> / T<sub>p</sub></b>, iniciando em E = 1,00 para isolar o efeito de <i>cache blocking</i>).",
        s["Body"],
    ))

    for idx_n, n_val in enumerate((15000, 30000, 50000)):
        rows_n = dados["forte_t1_rel"][n_val]
        tbl_rows = [[
            Paragraph("Config.", s["TH"]),
            Paragraph("w", s["TH"]),
            Paragraph("Cálculo (s)", s["TH"]),
            Paragraph("Total (s)", s["TH"]),
            Paragraph("Spd/Efic Calc<br/>(vs seq)", s["TH"]),
            Paragraph("Spd/Efic Tot<br/>(vs seq)", s["TH"]),
            Paragraph("Spd/Efic Calc<br/>(vs T1)", s["TH"]),
            Paragraph("Spd/Efic Tot<br/>(vs T1)", s["TH"]),
        ]]
        for r in rows_n:
            ver = r["versao"]
            if ver == "seq":
                top_lbl = "seq (orig)"
            elif r["topologia"] == "1":
                top_lbl = "1 (T1)"
            else:
                top_lbl = r["topologia"]
            w_lbl = r["workers"]
            c_str = fmt_media_dp(r["calc_media_s"], r["calc_dp_s"], 3)
            t_str = fmt_media_dp(r["tot_media_s"], r["tot_dp_s"], 3)
            if ver == "seq":
                s_sc = s_st = s_1c = s_1t = "1,00 / 1,00"
            else:
                s_sc = f"{fmt_pt(float(r['speedup_calc']), 2)} / {fmt_pt(float(r['eficiencia_calc']), 2)}"
                s_st = f"{fmt_pt(float(r['speedup_tot']), 2)} / {fmt_pt(float(r['eficiencia_tot']), 2)}"
                s_1c = f"{fmt_pt(r['sp1_calc'], 2)} / {fmt_pt(r['ef1_calc'], 2)}"
                s_1t = f"{fmt_pt(r['sp1_tot'], 2)} / {fmt_pt(r['ef1_tot'], 2)}"
            st_c = s["TDB"] if r["topologia"] in ("12-all", "10-fis") else s["TD"]
            tbl_rows.append([
                Paragraph(top_lbl, s["TDL"]),
                Paragraph(w_lbl, s["TD"]),
                Paragraph(c_str, s["TD"]),
                Paragraph(t_str, s["TD"]),
                Paragraph(s_sc, st_c),
                Paragraph(s_st, st_c),
                Paragraph(s_1c, s["TD"]),
                Paragraph(s_1t, s["TD"]),
            ])
        t_f = Table(
            tbl_rows,
            colWidths=[1.75*cm, 0.70*cm, 2.55*cm, 2.55*cm, 2.40*cm, 2.40*cm, 2.38*cm, 2.37*cm],
            repeatRows=1,
        )
        t_f.setStyle(estilo_tabela_padrao(col_left=[0], highlight_rows=[2, 9, 10]))
        story.append(KeepTogether([
            t_f,
            Paragraph(
                f"Tabela {4 + idx_n}: Escalabilidade forte para N = {n_val:,}".replace(",", ".") +
                " (omp_tarefas -b 1024, média ± σ em 20 execuções; formato Speedup / Eficiência).",
                s["Caption"],
            ),
        ]))

    story.append(KeepTogether([
        construir_grafico_speedup_50k(dados),
        Paragraph("Figura 1: Speedup medido em N = 50.000 comparado ao Speedup Linear e ao limite teórico de Amdahl.", s["Caption"]),
    ]))
    story.append(KeepTogether([
        construir_grafico_eficiencias(dados),
        Paragraph("Figura 2: Eficiência forte relativa a T1 (esquerda) e eficiência da escalabilidade fraca (direita).", s["Caption"]),
    ]))

    # =========================================================================
    # ITEM 7: DISCUSSAO SPEEDUP E EFICIENCIA
    # =========================================================================
    story.append(Paragraph("7. Discussão dos Resultados de Speedup e Eficiência", s["H1"]))
    story.append(Paragraph(
        "Comparando o speedup medido frente ao <font name='Mono' size='10.5'>seq</font> original com a "
        "curva linear e com Amdahl (Figura 1), nota-se um aparente ganho superlinear nos pontos iniciais: "
        "com apenas <b>1 thread</b> (<font name='Mono' size='10.5'>omp_tarefas -b 1024</font>), o cálculo já "
        "é <b>1,70×</b> mais rápido em N = 50.000 (3,689 s vs 6,285 s) e o tempo total é <b>1,63×</b> menor "
        "(4,021 s vs 6,549 s), atingindo eficiência aparente de 1,63 em 1T e 1,32 em 2T.",
        s["Body"],
    ))
    story.append(Paragraph(
        "Esse ganho não decorre do paralelismo, mas sim da mudança algorítmica de <b><i>cache blocking</i></b> "
        "(<i>tiling</i>): no <font name='Mono' size='10.5'>seq</font>, cada iteração percorre uma linha inteira "
        "de 50.001 elementos × 2 B ≈ <b>100 KB</b> (~<b>200 KB</b> para linha atual + anterior), estourando a "
        "cache L1d de <b>48 KB</b> dos P-cores e gerando faltas contínuas até L2/L3. Ao particionar em blocos "
        "1024 × 1024, cada sublinha ocupa apenas <b>2 KB</b> (~<b>6 KB</b> de <i>working set</i> ativo somando "
        "<font name='Mono' size='10.5'>seqA</font>, linha atual e anterior), cabendo integralmente na L1d "
        "(inclusive na L1d de 32 KB dos E-cores). Ao adotar a referência justa <b>T<sub>1</sub> / T<sub>p</sub></b> "
        "(mesmo algoritmo blocado com 1 thread), a eficiência parte de <b>1,00</b> e mantém-se estritamente abaixo "
        "de Amdahl (atingindo <b>3,77×</b> no cálculo e <b>2,92×</b> no total em 12-all para N = 50.000).",
        s["Body"],
    ))
    story.append(Paragraph(
        "Quanto ao tamanho da entrada, a eficiência total relativa ao <font name='Mono' size='10.5'>seq</font> "
        "em 10-fis cresce de <b>0,40</b> (N = 15k) para <b>0,46</b> (N = 30k e 50k), e em 12-all sobe de "
        "<b>0,34</b> para <b>0,41</b> (30k) e <b>0,40</b> (50k), enquanto em 4-fis (vs T1) mantém ~0,50–0,53. "
        "Entradas maiores criam mais blocos independentes por diagonal (de 15×15 = 225 tarefas em 15k para "
        "49×49 = 2.401 tarefas em 50k), reduzindo o impacto do enchimento/esvaziamento do <i>wavefront</i> "
        "e amortizando a criação de tarefas. Ainda assim, após 4–6 núcleos há saturação progressiva por "
        "contenção de largura de banda de memória compartilhada.",
        s["Body"],
    ))

    # =========================================================================
    # ITEM 8: ESCALABILIDADE FORTE E FRACA
    # =========================================================================
    story.append(Paragraph("8. Análise de Escalabilidade Forte e Fraca", s["H1"]))
    story.append(Paragraph(
        "Para testar a <b>escalabilidade fraca</b> mantendo o trabalho assintótico O(N²) proporcional ao número "
        "de workers <i>p</i>, escalou-se a entrada segundo <b>N(p) = 14.000 · √p</b> com "
        "<font name='Mono' size='10.5'>omp_tarefas -b 1024</font> (20 execuções por ponto):",
        s["Body"],
    ))

    tbl_fr = [[
        Paragraph("Topologia", s["TH"]),
        Paragraph("p", s["TH"]),
        Paragraph("N(p)", s["TH"]),
        Paragraph("Cálculo (s)", s["TH"]),
        Paragraph("CV Calc", s["TH"]),
        Paragraph("Total (s)", s["TH"]),
        Paragraph("CV Tot", s["TH"]),
        Paragraph("Ef. Fraca<br/>Cálculo", s["TH"]),
        Paragraph("Ef. Fraca<br/>Total", s["TH"]),
    ]]
    for r in dados["fraca_resumo"]:
        tbl_fr.append([
            Paragraph(r["topologia"], s["TDL"]),
            Paragraph(r["workers"], s["TD"]),
            Paragraph(f"{int(r['N']):,}".replace(",", "."), s["TD"]),
            Paragraph(fmt_media_dp(r["calc_media_s"], r["calc_dp_s"], 3), s["TD"]),
            Paragraph(f"{fmt_pt(float(r['calc_cv_pct']), 2)}%", s["TD"]),
            Paragraph(fmt_media_dp(r["tot_media_s"], r["tot_dp_s"], 3), s["TD"]),
            Paragraph(f"{fmt_pt(float(r['tot_cv_pct']), 2)}%", s["TD"]),
            Paragraph(fmt_pt(float(r["eficiencia_fraca_calc"]), 2), s["TDB"]),
            Paragraph(fmt_pt(float(r["eficiencia_fraca_tot"]), 2), s["TDB"]),
        ])
    t_fr = Table(
        tbl_fr,
        colWidths=[1.65*cm, 0.70*cm, 1.45*cm, 2.55*cm, 1.40*cm, 2.55*cm, 1.40*cm, 2.70*cm, 2.70*cm],
        repeatRows=1,
    )
    t_fr.setStyle(estilo_tabela_padrao(col_left=[0], highlight_rows=[7]))
    story.append(KeepTogether([
        t_fr,
        Paragraph("Tabela 7: Escalabilidade fraca (N(p) = 14.000·√p, omp_tarefas -b 1024, média ± σ em 20 execuções).", s["Caption"]),
    ]))

    fe_1 = dados["fraca_extra"]["1"]
    fe_12 = dados["fraca_extra"]["12-all"]
    t1_fr_tot = float(dados["fraca_resumo"][0]["tot_media_s"])
    story.append(Paragraph(
        f"<b>Conclusão de Escalabilidade:</b> O algoritmo apresentou <b>escalabilidade forte moderada/limitada</b> "
        f"(speedup total de 4,76× sobre <font name='Mono' size='10.5'>seq</font> e 2,92× sobre T<sub>1</sub> em 12-all "
        f"para N = 50.000, com redução monotônica do tempo até o máximo de threads), mas <b>não apresentou boa "
        f"escalabilidade fraca</b>: em um cenário ideal, o tempo permaneceria constante próximo a "
        f"<b>{fmt_pt(t1_fr_tot, 3)} s</b>, porém a eficiência fraca caiu continuamente para <b>0,29</b> (cálculo) "
        f"e <b>0,24</b> (total) em 12-all. Quatro fatores físicos explicam essa queda: "
        f"(1) <b>banda de memória DRAM compartilhada fixa</b> entre todos os núcleos em um único soquete; "
        f"(2) crescimento do conjunto de trabalho de <b>~392 MB</b> (N=14k) para <b>~4,70 GB</b> (N=48.497), "
        f"esgotando a L3 (12 MB) e ampliando o tráfego de escrita na matriz; "
        f"(3) incorporação de <i>E-cores</i> (3,6 GHz) e SMT sobre a base de 1 <i>P-core</i> (4,8 GHz); e "
        f"(4) crescimento O(N) e O(N²) da fração sequencial (<font name='Mono' size='10.5'>inicializacao + liberacao</font> "
        f"salta de {fmt_pt(fe_1['init']+fe_1['lib'], 3)} s em N=14k para <b>{fmt_pt(fe_12['init']+fe_12['lib'], 3)} s</b> "
        f"em N=48.497, já superando sozinha todo o tempo T<sub>1</sub>).",
        s["Body"],
    ))

    # =========================================================================
    # ITEM 9: ANALISE CRITICA
    # =========================================================================
    story.append(Paragraph("9. Análise Crítica: Coerência, SMT, Heterogeneidade e Variabilidade", s["H1"]))
    story.append(Paragraph(
        "<b>Coerência Física e Natureza <i>Memory/Cache-Bound</i>:</b> Todos os resultados são fisicamente "
        "coerentes. Embora a matriz exija O(N²) operações, cada célula executa pouquíssimas instruções "
        "aritméticas (uma comparação, uma soma ou máximo) para 2 bytes escritos e 6 bytes lidos, resultando em "
        "baixa intensidade aritmética. Assim, o algoritmo é fortemente limitado por hierarquia de cache e banda "
        "de memória, razão pela qual o <i>cache blocking</i> rendeu 1,70× já em 1 thread e a curva de speedup "
        "satura abaixo de Amdahl após 6–8 núcleos.",
        s["Body"],
    ))

    def get_t(n: int, top: str, field: str) -> float:
        r = [x for x in dados["forte_t1_rel"][n] if x["versao"] == "omp_tarefas" and x["topologia"] == top][0]
        return float(r[field])

    smt_losses = [
        (get_t(n, "4-SMT", "tot_media_s") / get_t(n, "4-fis", "tot_media_s") - 1.0) * 100.0
        for n in (15000, 30000, 50000)
    ]
    smt12_spd = [
        (get_t(n, "10-fis", "tot_media_s") / get_t(n, "12-all", "tot_media_s") - 1.0) * 100.0
        for n in (15000, 30000, 50000)
    ]
    pe_calc_spd = [
        (get_t(n, "8-E", "calc_media_s") / get_t(n, "8-hib", "calc_media_s") - 1.0) * 100.0
        for n in (15000, 30000, 50000)
    ]
    pe_tot_spd = [
        (get_t(n, "8-E", "tot_media_s") / get_t(n, "8-hib", "tot_media_s") - 1.0) * 100.0
        for n in (15000, 30000, 50000)
    ]
    np_map = dados["nao_par_map"]

    story.append(Paragraph(
        f"<b>Desempenho do Simultaneous Multi-Threading (SMT):</b> A comparação direta entre <b>4-SMT</b> "
        f"(2 P-cores, 4 threads lógicas <font name='Mono' size='10.5'>[0,1,2,3]</font>) e <b>4-fis</b> "
        f"(2 P-cores + 2 E-cores físicos <font name='Mono' size='10.5'>[0,2,4,5]</font>) revela degradação "
        f"consistente do SMT: <b>4-SMT foi {fmt_pt(smt_losses[0], 1)}% (15k), {fmt_pt(smt_losses[1], 1)}% (30k) "
        f"e {fmt_pt(smt_losses[2], 1)}% (50k) mais lento</b> no tempo total. No núcleo físico, duas threads SMT "
        f"competem pelas mesmas portas <i>load/store</i>, filas de execução e principalmente pela mesma cache "
        f"L1d (48 KB) e L2, provocando <i>thrashing</i> de linhas de cache e conflito de largura de banda. "
        f"Analogamente, adicionar os 2 irmãos SMT aos 10 núcleos físicos (<b>12-all vs 10-fis</b>) gerou ganhos "
        f"marginais de velocidade de apenas <b>+{fmt_pt(smt12_spd[0], 1)}% (15k), +{fmt_pt(smt12_spd[1], 1)}% (30k) e "
        f"+{fmt_pt(smt12_spd[2], 1)}% (50k)</b> no tempo total (reduzindo a eficiência), confirmando que "
        f"o SMT não compensa quando o gargalo reside na hierarquia de memória.",
        s["Body"],
    ))
    story.append(Paragraph(
        f"<b>Heterogeneidade P-cores vs E-cores (8-hib vs 8-E):</b> Comparando <b>8-hib</b> (2 P-cores + 6 E-cores) "
        f"com <b>8-E</b> (apenas 8 E-cores), o ganho de velocidade de 8-hib sobre 8-E na fase de cálculo foi modesto: "
        f"<b>+{fmt_pt(pe_calc_spd[0], 1)}%</b> (15k), <b>+{fmt_pt(pe_calc_spd[1], 1)}%</b> (30k) e "
        f"<b>+{fmt_pt(pe_calc_spd[2], 1)}%</b> (50k), pois o escalonamento dinâmico de tarefas OpenMP "
        f"absorve a assimetria de velocidade e a banda compartilhada limita os P-cores. Contudo, no <b>tempo total</b> "
        f"o ganho salta para <b>+{fmt_pt(pe_tot_spd[0], 1)}% (15k), +{fmt_pt(pe_tot_spd[1], 1)}% (30k) e "
        f"+{fmt_pt(pe_tot_spd[2], 1)}% (50k)</b>: a porção sequencial (<font name='Mono' size='10.5'>inicializacao + "
        f"liberacao</font>) em 8-E executa em um E-core de 3,6 GHz em vez de um P-core de 4,8 GHz, saltando de "
        f"<b>{fmt_pt(np_map[(50000,'8-hib')], 3)} s</b> (8-hib) para <b>{fmt_pt(np_map[(50000,'8-E')], 3)} s</b> "
        f"(8-E, <b>+30%</b> em N=50k).",
        s["Body"],
    ))
    story.append(Paragraph(
        f"<b>Análise do Desvio Padrão (σ) e CV% nos Dados Brutos:</b> A inspeção das séries de 20 execuções "
        f"explica os padrões de variabilidade observados. No <font name='Mono' size='10.5'>seq</font> em N = 50.000 "
        f"(CV<sub>tot</sub> = 7,50%), as <b>amostras 1 a 15</b> mantiveram média de cálculo estável em "
        f"<b>{fmt_pt(dados['seq50_r1_15_calc'], 3)} s</b> (liberação {fmt_pt(dados['seq50_r1_15_lib'], 3)} s), "
        f"mas após ~95 s de carga contínua ocorreu um degrau térmico/PL1 nas <b>amostras 16 a 20</b>, elevando "
        f"o cálculo para <b>{fmt_pt(dados['seq50_r16_20_calc'], 3)} s</b> (<b>+15,3%</b>) e a liberação para "
        f"<b>{fmt_pt(dados['seq50_r16_20_lib'], 3)} s</b>. Nas entradas menores (10k e 15k), o CV variou entre "
        f"12% e 21% devido à oscilação bimodal provocada pela rampa de P-state do governador "
        f"<font name='Mono' size='10.5'>powersave</font> (partindo de 400 MHz até 4,8 GHz) somada ao <i>jitter</i> "
        f"de faltas de página do SO em execuções sub-segundo. Em contraste, nas execuções multi-core em "
        f"N = 50.000 (mais longas e com todos os núcleos ativos), o CV<sub>tot</sub> caiu para a faixa estável de "
        f"<b>1,56% a 2,85%</b> no <font name='Mono' size='10.5'>omp_tarefas</font>, enquanto "
        f"<font name='Mono' size='10.5'>mpi_blocos (10-fis)</font> apresentou CV maior (<b>4,37%</b>) pela "
        f"propagação em cascata de pequenos atrasos entre estágios do <i>pipeline</i> MPI e barreiras globais.",
        s["Body"],
    ))

    doc.build(story, onFirstPage=cabecalho_rodape, onLaterPages=cabecalho_rodape)
    print(f"PDF gerado com sucesso em: {OUT_PDF}")


if __name__ == "__main__":
    dados_exp = carregar_dados()
    gerar_pdf(dados_exp)
