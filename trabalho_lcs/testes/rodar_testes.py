#!/usr/bin/env python3
"""Suite de testes e edge cases para o LCS paralelo (OpenMP + MPI)."""

import argparse
import concurrent.futures
from dataclasses import dataclass
import difflib
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple


DIR_TRABALHO = Path(__file__).resolve().parent.parent
DIR_ORIGINAIS = DIR_TRABALHO.parent / "lcs_source_code"

BIN_PARALELO = DIR_TRABALHO / "lcs_paralelo"
BIN_PARALELO_DEBUG = DIR_TRABALHO / "lcs_paralelo_debug"
BIN_REF = DIR_TRABALHO / "lcs_ref"
BIN_REF_DEBUG = DIR_TRABALHO / "lcs_ref_debug"

VERSOES_CONHECIDAS = (
    "seq",
    "omp_diag",
    "omp_blocos",
    "omp_tarefas",
    "mpi_blocos",
    "hibrido",
    "auto",
)


@dataclass(frozen=True)
class CasoTeste:
    """Representa um caso de teste de entrada para o LCS."""

    nome: str
    conteudo_a: Optional[bytes] = None
    conteudo_b: Optional[bytes] = None
    arquivo_externo_a: Optional[Path] = None
    arquivo_externo_b: Optional[Path] = None
    usar_flags_arquivo: bool = True
    score_esperado: Optional[int] = None
    verificar_matriz: bool = True
    verificar_medicao_tempo: bool = False
    repeticoes_minimas: int = 1
    essencial: bool = False


@dataclass(frozen=True)
class ReferenciaCaso:
    """Armazena as saidas de referencia pre-computadas para um caso de teste."""

    stdout_normal: str
    stdout_debug: Optional[str] = None


@dataclass(frozen=True)
class ConfigExecucao:
    """Define uma configuracao de execucao para uma versao do algoritmo."""

    versao: str
    tam_bloco: Optional[int] = None
    num_threads: Optional[int] = None
    num_procs: Optional[int] = None

    def descricao(self) -> str:
        partes = ["v=auto(sem -v)" if self.versao == "auto" else f"v={self.versao}"]
        if self.tam_bloco is not None:
            partes.append(f"b={self.tam_bloco}")
        elif self.versao in ("omp_blocos", "omp_tarefas", "mpi_blocos", "hibrido", "auto"):
            partes.append("b=padrao")
        if self.num_threads is not None:
            partes.append(f"t={self.num_threads}")
        if self.num_procs is not None:
            partes.append(f"np={self.num_procs}")
        return ", ".join(partes)


def gerar_sequencia_aleatoria(semente: int, tamanho: int, alfabeto: str) -> str:
    """Gera uma sequencia deterministica a partir de uma semente e alfabeto."""
    rng = random.Random(semente)
    return "".join(rng.choice(alfabeto) for _ in range(tamanho))


def construir_casos_teste() -> List[CasoTeste]:
    """Constroi a lista completa de casos de teste e edge cases."""
    arq_orig_a = DIR_ORIGINAIS / "fileA.in"
    arq_orig_b = DIR_ORIGINAIS / "fileB.in"

    alfa_dna = "ACGT"
    alfa_misto = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    casos: List[CasoTeste] = [
        # Exemplo do enunciado sem -a/-B (leitura padrao de fileA.in e fileB.in no diretorio)
        CasoTeste(
            nome="enunciado_padrao_sem_flags",
            conteudo_a=arq_orig_a.read_bytes(),
            conteudo_b=arq_orig_b.read_bytes(),
            usar_flags_arquivo=False,
            score_esperado=5,
            verificar_matriz=True,
            verificar_medicao_tempo=True,
            essencial=True,
        ),
        # Exemplo do enunciado passando explicitamente -a e -B
        CasoTeste(
            nome="enunciado_com_flags_a_B",
            arquivo_externo_a=arq_orig_a,
            arquivo_externo_b=arq_orig_b,
            usar_flags_arquivo=True,
            score_esperado=5,
            verificar_matriz=True,
            verificar_medicao_tempo=True,
        ),
        # Entradas vazias
        CasoTeste(
            nome="vazio_0x0",
            conteudo_a=b"",
            conteudo_b=b"",
            score_esperado=0,
            verificar_matriz=True,
            verificar_medicao_tempo=True,
            essencial=True,
        ),
        CasoTeste(
            nome="vazio_0x15",
            conteudo_a=b"",
            conteudo_b=b"ABCDEFGHIJKLMNO\n",
            score_esperado=0,
            verificar_matriz=True,
            essencial=True,
        ),
        CasoTeste(
            nome="vazio_15x0",
            conteudo_a=b"ABCDEFGHIJKLMNO\n",
            conteudo_b=b"",
            score_esperado=0,
            verificar_matriz=True,
            essencial=True,
        ),
        CasoTeste(
            nome="apenas_quebra_linha",
            conteudo_a=b"\n",
            conteudo_b=b"ABC\n",
            score_esperado=0,
            verificar_matriz=True,
        ),
        # Unitarios
        CasoTeste(
            nome="unitario_1x1_igual",
            conteudo_a=b"A\n",
            conteudo_b=b"A\n",
            score_esperado=1,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="unitario_1x1_diferente",
            conteudo_a=b"A\n",
            conteudo_b=b"B\n",
            score_esperado=0,
            verificar_matriz=True,
        ),
        # Pequenas
        CasoTeste(
            nome="pequena_3x4",
            conteudo_a=b"ABC\n",
            conteudo_b=b"ACBD\n",
            score_esperado=2,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="pequena_7x7",
            conteudo_a=(gerar_sequencia_aleatoria(7, 7, alfa_dna) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(77, 7, alfa_dna) + "\n").encode(),
            verificar_matriz=True,
            essencial=True,
        ),
        CasoTeste(
            nome="pequena_9x2",
            conteudo_a=(gerar_sequencia_aleatoria(9, 9, alfa_dna) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(2, 2, alfa_dna) + "\n").encode(),
            verificar_matriz=True,
            essencial=True,
        ),
        CasoTeste(
            nome="pequena_10x10",
            conteudo_a=(gerar_sequencia_aleatoria(10, 10, alfa_dna) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(11, 10, alfa_dna) + "\n").encode(),
            verificar_matriz=True,
        ),
        # 1D: 1x50 e 50x1 (inicio, meio, fim, ausente)
        CasoTeste(
            nome="1d_1x50_match_inicio",
            conteudo_a=b"A\n",
            conteudo_b=("A" + "B" * 49 + "\n").encode(),
            score_esperado=1,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="1d_1x50_match_meio",
            conteudo_a=b"A\n",
            conteudo_b=("B" * 25 + "A" + "C" * 24 + "\n").encode(),
            score_esperado=1,
            verificar_matriz=True,
            essencial=True,
        ),
        CasoTeste(
            nome="1d_1x50_match_fim",
            conteudo_a=b"A\n",
            conteudo_b=("B" * 49 + "A\n").encode(),
            score_esperado=1,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="1d_1x50_match_ausente",
            conteudo_a=b"Z\n",
            conteudo_b=("AB" * 25 + "\n").encode(),
            score_esperado=0,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="1d_50x1_match_inicio",
            conteudo_a=("A" + "B" * 49 + "\n").encode(),
            conteudo_b=b"A\n",
            score_esperado=1,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="1d_50x1_match_meio",
            conteudo_a=("B" * 25 + "A" + "C" * 24 + "\n").encode(),
            conteudo_b=b"A\n",
            score_esperado=1,
            verificar_matriz=True,
            essencial=True,
        ),
        CasoTeste(
            nome="1d_50x1_match_fim",
            conteudo_a=("B" * 49 + "A\n").encode(),
            conteudo_b=b"A\n",
            score_esperado=1,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="1d_50x1_match_ausente",
            conteudo_a=("AB" * 25 + "\n").encode(),
            conteudo_b=b"Z\n",
            score_esperado=0,
            verificar_matriz=True,
        ),
        # 100% identicos e 100% disjuntos
        CasoTeste(
            nome="identicos_40x40",
            conteudo_a=(gerar_sequencia_aleatoria(40, 40, alfa_misto) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(40, 40, alfa_misto) + "\n").encode(),
            score_esperado=40,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="disjuntos_40x40",
            conteudo_a=("ABCDE" * 8 + "\n").encode(),
            conteudo_b=("FGHIJ" * 8 + "\n").encode(),
            score_esperado=0,
            verificar_matriz=True,
        ),
        # Caractere unico repetido
        CasoTeste(
            nome="caractere_unico_repetido_30x20",
            conteudo_a=("X" * 30 + "\n").encode(),
            conteudo_b=("X" * 20 + "\n").encode(),
            score_esperado=20,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="caractere_unico_diferente_20x20",
            conteudo_a=("X" * 20 + "\n").encode(),
            conteudo_b=("Y" * 20 + "\n").encode(),
            score_esperado=0,
            verificar_matriz=True,
        ),
        # Maiusculas, minusculas e numeros (sensivel a caso)
        CasoTeste(
            nome="alfanumerico_case_sensitive_64x64",
            conteudo_a=(gerar_sequencia_aleatoria(64, 64, alfa_misto) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(65, 64, alfa_misto) + "\n").encode(),
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="maiusculas_vs_minusculas_10x10",
            conteudo_a=b"ABCDEFGHIJ\n",
            conteudo_b=b"abcdefghij\n",
            score_esperado=0,
            verificar_matriz=True,
        ),
        # Com e sem \n final, e com multiplas quebras de linha
        CasoTeste(
            nome="sem_newline_final",
            conteudo_a=b"heagawghee",
            conteudo_b=b"pawheae",
            score_esperado=5,
            verificar_matriz=True,
        ),
        CasoTeste(
            nome="multiplos_newlines_internos",
            conteudo_a=b"heag\nawg\nhee\n",
            conteudo_b=b"paw\nhea\ne",
            score_esperado=5,
            verificar_matriz=True,
        ),
        # Dimensoes primas
        CasoTeste(
            nome="primo_17x31",
            conteudo_a=(gerar_sequencia_aleatoria(17, 17, alfa_dna) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(31, 31, alfa_dna) + "\n").encode(),
            verificar_matriz=True,
            essencial=True,
        ),
        CasoTeste(
            nome="primo_101x97",
            conteudo_a=(gerar_sequencia_aleatoria(101, 101, alfa_misto) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(97, 97, alfa_misto) + "\n").encode(),
            verificar_matriz=True,
            verificar_medicao_tempo=True,
            repeticoes_minimas=5,
            essencial=True,
        ),
        CasoTeste(
            nome="primo_503x257",
            conteudo_a=(gerar_sequencia_aleatoria(503, 503, alfa_misto) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(257, 257, alfa_misto) + "\n").encode(),
            verificar_matriz=False,
            essencial=True,
        ),
        # Retangulares extremas
        CasoTeste(
            nome="retangular_5x500",
            conteudo_a=(gerar_sequencia_aleatoria(5, 5, alfa_dna) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(500, 500, alfa_dna) + "\n").encode(),
            verificar_matriz=True,
            essencial=True,
        ),
        CasoTeste(
            nome="retangular_500x5",
            conteudo_a=(gerar_sequencia_aleatoria(500, 500, alfa_dna) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(5, 5, alfa_dna) + "\n").encode(),
            verificar_matriz=True,
            essencial=True,
        ),
        # Sequencia maior
        CasoTeste(
            nome="grande_1000x1000",
            conteudo_a=(gerar_sequencia_aleatoria(1000, 1000, alfa_dna) + "\n").encode(),
            conteudo_b=(gerar_sequencia_aleatoria(1001, 1000, alfa_dna) + "\n").encode(),
            verificar_matriz=False,
            verificar_medicao_tempo=True,
        ),
    ]
    return casos


def extrair_score(saida_stdout: str) -> Optional[int]:
    """Extrai o score se o stdout seguir exatamente '\\nScore: %d\\n'."""
    match = re.fullmatch(r"\nScore: (\d+)\n", saida_stdout)
    if match is None:
        return None
    return int(match.group(1))


def executar_comando(
    comando: List[str],
    cwd: Path,
    env_extra: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Executa um binario capturando stdout e stderr como texto."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        comando,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def gerar_diff_resumido(esperado: str, obtido: str, max_linhas: int = 20) -> str:
    """Retorna um diff unificado curto para diagnostico de falhas."""
    linhas = list(
        difflib.unified_diff(
            esperado.splitlines(keepends=True),
            obtido.splitlines(keepends=True),
            fromfile="referencia",
            tofile="paralelo",
            n=2,
        )
    )
    if len(linhas) > max_linhas:
        linhas = linhas[:max_linhas] + [f"... ({len(linhas) - max_linhas} linhas omitidas)\n"]
    return "".join(linhas)


def montar_comando_paralelo(
    binario: Path,
    cfg: ConfigExecucao,
    arq_a: Optional[Path],
    arq_b: Optional[Path],
    usar_flags_arquivo: bool,
    medir_tempos: bool,
    oversubscribe: bool,
) -> Tuple[List[str], Dict[str, str]]:
    """Monta a linha de comando e variaveis de ambiente para o binario paralelo."""
    cmd: List[str] = []
    if cfg.num_procs is not None:
        cmd.extend(["mpirun"])
        if oversubscribe:
            cmd.append("--oversubscribe")
        cmd.extend(["-np", str(cfg.num_procs)])

    cmd.append(str(binario))
    if cfg.versao != "auto":
        cmd.extend(["-v", cfg.versao])
    if cfg.tam_bloco is not None:
        cmd.extend(["-b", str(cfg.tam_bloco)])
    if usar_flags_arquivo and arq_a is not None and arq_b is not None:
        cmd.extend(["-a", str(arq_a), "-B", str(arq_b)])
    if medir_tempos:
        cmd.append("-m")

    env_extra: Dict[str, str] = {}
    if cfg.num_threads is not None:
        env_extra["OMP_NUM_THREADS"] = str(cfg.num_threads)
    return cmd, env_extra


def preparar_entradas_referencia(caso: CasoTeste, dir_ref: Path) -> None:
    """Escreve fileA.in e fileB.in no diretorio temporario da referencia."""
    if caso.arquivo_externo_a is not None and caso.arquivo_externo_b is not None:
        shutil.copyfile(caso.arquivo_externo_a, dir_ref / "fileA.in")
        shutil.copyfile(caso.arquivo_externo_b, dir_ref / "fileB.in")
        return
    assert caso.conteudo_a is not None and caso.conteudo_b is not None
    (dir_ref / "fileA.in").write_bytes(caso.conteudo_a)
    (dir_ref / "fileB.in").write_bytes(caso.conteudo_b)


def preparar_entradas_paralelo(
    caso: CasoTeste,
    dir_par: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Prepara os arquivos de entrada no diretorio temporario do binario paralelo."""
    if caso.arquivo_externo_a is not None and caso.arquivo_externo_b is not None:
        if caso.usar_flags_arquivo:
            # Cria arquivos locais diferentes para garantir que -a/-B realmente sobrescrevam o padrao
            (dir_par / "fileA.in").write_bytes(b"ZZZZ\n")
            (dir_par / "fileB.in").write_bytes(b"WWWW\n")
            return caso.arquivo_externo_a, caso.arquivo_externo_b
        shutil.copyfile(caso.arquivo_externo_a, dir_par / "fileA.in")
        shutil.copyfile(caso.arquivo_externo_b, dir_par / "fileB.in")
        return dir_par / "fileA.in", dir_par / "fileB.in"

    assert caso.conteudo_a is not None and caso.conteudo_b is not None
    arq_par_a = dir_par / ("entrada_A.in" if caso.usar_flags_arquivo else "fileA.in")
    arq_par_b = dir_par / ("entrada_B.in" if caso.usar_flags_arquivo else "fileB.in")
    arq_par_a.write_bytes(caso.conteudo_a)
    arq_par_b.write_bytes(caso.conteudo_b)
    return arq_par_a, arq_par_b


def gerar_referencias(casos: List[CasoTeste]) -> Dict[str, ReferenciaCaso]:
    """Executa lcs_ref e lcs_ref_debug uma unica vez por caso de teste."""
    referencias: Dict[str, ReferenciaCaso] = {}
    with tempfile.TemporaryDirectory(prefix="lcs_refs_") as tmp_dir:
        base_tmp = Path(tmp_dir)
        for caso in casos:
            dir_ref = base_tmp / caso.nome
            dir_ref.mkdir()
            preparar_entradas_referencia(caso, dir_ref)

            res_ref = executar_comando([str(BIN_REF)], cwd=dir_ref)
            if res_ref.returncode != 0:
                raise RuntimeError(
                    f"lcs_ref falhou em '{caso.nome}' (codigo={res_ref.returncode}): {res_ref.stderr}"
                )
            score_ref = extrair_score(res_ref.stdout)
            if score_ref is None:
                raise RuntimeError(
                    f"Formato inesperado em lcs_ref stdout para '{caso.nome}': {res_ref.stdout!r}"
                )
            if caso.score_esperado is not None and score_ref != caso.score_esperado:
                raise RuntimeError(
                    f"Score de lcs_ref ({score_ref}) difere do esperado ({caso.score_esperado}) em '{caso.nome}'"
                )

            stdout_debug = None
            if caso.verificar_matriz:
                res_dbg = executar_comando([str(BIN_REF_DEBUG)], cwd=dir_ref)
                if res_dbg.returncode != 0:
                    raise RuntimeError(
                        f"lcs_ref_debug falhou em '{caso.nome}' (codigo={res_dbg.returncode}): {res_dbg.stderr}"
                    )
                stdout_debug = res_dbg.stdout

            referencias[caso.nome] = ReferenciaCaso(
                stdout_normal=res_ref.stdout,
                stdout_debug=stdout_debug,
            )
    return referencias


def validar_caso(
    caso: CasoTeste,
    ref: ReferenciaCaso,
    cfg: ConfigExecucao,
    repeticoes: int,
    oversubscribe: bool,
) -> Optional[str]:
    """Executa um caso de teste contra a referencia e retorna None se passar ou mensagem de erro."""
    repeticoes_efetivas = max(repeticoes, caso.repeticoes_minimas if cfg.versao != "seq" else 1)

    with tempfile.TemporaryDirectory(prefix="lcst_") as tmp_dir:
        dir_par = Path(tmp_dir)
        dir_ompi = dir_par / "t"
        dir_ompi.mkdir()
        arq_par_a, arq_par_b = preparar_entradas_paralelo(caso, dir_par)

        # 1. Execucoes do binario paralelo (com repeticoes para detectar race conditions)
        for rep in range(1, repeticoes_efetivas + 1):
            cmd_par, env_par = montar_comando_paralelo(
                BIN_PARALELO,
                cfg,
                arq_par_a,
                arq_par_b,
                caso.usar_flags_arquivo,
                medir_tempos=False,
                oversubscribe=oversubscribe,
            )
            env_par["TMPDIR"] = str(dir_ompi)
            res_par = executar_comando(cmd_par, cwd=dir_par, env_extra=env_par)
            if res_par.returncode != 0:
                return f"[rep {rep}] lcs_paralelo falhou (codigo={res_par.returncode}): {res_par.stderr}"
            if res_par.stdout != ref.stdout_normal:
                return (
                    f"[rep {rep}] Divergencia de stdout sem DEBUGMATRIX: "
                    f"esperado {ref.stdout_normal!r}, obtido {res_par.stdout!r}"
                )
            if res_par.stderr != "":
                return f"[rep {rep}] stderr deveria estar vazio sem -m, obtido: {res_par.stderr!r}"

        # 2. Validacao da matriz completa com DEBUGMATRIX
        if caso.verificar_matriz:
            assert ref.stdout_debug is not None
            cmd_dbg, env_dbg = montar_comando_paralelo(
                BIN_PARALELO_DEBUG,
                cfg,
                arq_par_a,
                arq_par_b,
                caso.usar_flags_arquivo,
                medir_tempos=False,
                oversubscribe=oversubscribe,
            )
            env_dbg["TMPDIR"] = str(dir_ompi)
            res_par_dbg = executar_comando(cmd_dbg, cwd=dir_par, env_extra=env_dbg)
            if res_par_dbg.returncode != 0:
                return f"lcs_paralelo_debug falhou (codigo={res_par_dbg.returncode}): {res_par_dbg.stderr}"
            if res_par_dbg.stdout != ref.stdout_debug:
                diff = gerar_diff_resumido(ref.stdout_debug, res_par_dbg.stdout)
                return f"Divergencia na matriz DEBUGMATRIX:\n{diff}"

        # 3. Validacao de -m (tempos apenas em stderr, stdout estrito)
        if caso.verificar_medicao_tempo:
            cmd_m, env_m = montar_comando_paralelo(
                BIN_PARALELO,
                cfg,
                arq_par_a,
                arq_par_b,
                caso.usar_flags_arquivo,
                medir_tempos=True,
                oversubscribe=oversubscribe,
            )
            env_m["TMPDIR"] = str(dir_ompi)
            res_m = executar_comando(cmd_m, cwd=dir_par, env_extra=env_m)
            if res_m.returncode != 0:
                return f"lcs_paralelo -m falhou (codigo={res_m.returncode}): {res_m.stderr}"
            if res_m.stdout != ref.stdout_normal:
                return (
                    f"Com -m, stdout foi alterado: esperado {ref.stdout_normal!r}, "
                    f"obtido {res_m.stdout!r}"
                )
            if "TEMPOS " not in res_m.stderr or "total=" not in res_m.stderr:
                return f"Com -m, stderr nao contem tempos esperados: {res_m.stderr!r}"

    return None


def gerar_configuracoes(
    versao: str,
    blocos: List[int],
    threads: List[int],
    procs: List[int],
) -> List[ConfigExecucao]:
    """Gera a matriz de configuracoes apropriada para cada versao."""
    thread_ref = 4 if 4 in threads else threads[-1]
    proc_ref = 2 if 2 in procs else procs[-1]

    if versao == "seq":
        return [ConfigExecucao(versao="seq")]
    if versao == "omp_diag":
        return [ConfigExecucao(versao="omp_diag", num_threads=t) for t in threads]
    if versao in ("omp_blocos", "omp_tarefas"):
        configs = [
            ConfigExecucao(versao=versao, tam_bloco=None, num_threads=thread_ref)
        ]
        configs.extend(
            ConfigExecucao(versao=versao, tam_bloco=b, num_threads=t)
            for b in blocos
            for t in threads
        )
        return configs
    if versao == "mpi_blocos":
        configs = [
            ConfigExecucao(versao="mpi_blocos", tam_bloco=None, num_threads=1, num_procs=proc_ref)
        ]
        configs.extend(
            ConfigExecucao(versao="mpi_blocos", tam_bloco=b, num_threads=1, num_procs=p)
            for b in blocos
            for p in procs
        )
        return configs
    if versao == "hibrido":
        pares_hibridos = ((1, 1), (1, 3), (2, 2), (2, 3), (3, 2), (4, 2))
        blocos_hibrido = [b for b in blocos if b != 256] or list(blocos)
        configs = [
            ConfigExecucao(versao="hibrido", tam_bloco=None, num_threads=2, num_procs=2),
            ConfigExecucao(versao="hibrido", tam_bloco=None, num_threads=2, num_procs=5),
        ]
        configs.extend(
            ConfigExecucao(versao="hibrido", tam_bloco=b, num_threads=t, num_procs=p)
            for b in blocos_hibrido
            for p, t in pares_hibridos
        )
        configs.extend(
            ConfigExecucao(versao="hibrido", tam_bloco=b, num_threads=2, num_procs=5)
            for b in blocos_hibrido
        )
        return configs
    if versao == "auto":
        return [
            ConfigExecucao(versao="auto", tam_bloco=None, num_threads=4, num_procs=None),
            ConfigExecucao(versao="auto", tam_bloco=None, num_threads=2, num_procs=2),
        ]
    return [ConfigExecucao(versao=versao)]


def dimensoes_caso(caso: CasoTeste) -> Tuple[int, int]:
    """Retorna os tamanhos (tamA, tamB) do caso de teste ignorando quebras de linha."""
    if caso.arquivo_externo_a is not None and caso.arquivo_externo_b is not None:
        dados_a = caso.arquivo_externo_a.read_bytes()
        dados_b = caso.arquivo_externo_b.read_bytes()
    else:
        assert caso.conteudo_a is not None and caso.conteudo_b is not None
        dados_a = caso.conteudo_a
        dados_b = caso.conteudo_b
    return len(dados_a.replace(b"\n", b"")), len(dados_b.replace(b"\n", b""))


def selecionar_casos_para_config(
    casos: List[CasoTeste],
    cfg: ConfigExecucao,
    blocos: List[int],
    threads: List[int],
    procs: List[int],
    modo: str,
) -> List[CasoTeste]:
    """Seleciona os casos aplicaveis conforme o modo, a configuracao e limites de granularidade."""
    # Para P=5, T=2 no modo hibrido, foca explicitamente nos casos tamB < P (e vazios)
    if cfg.versao == "hibrido" and cfg.num_procs == 5:
        selecionados = [
            caso
            for caso in casos
            if (dimensoes_caso(caso)[1] < 5 or dimensoes_caso(caso)[0] == 0)
            and (modo == "completo" or caso.essencial or caso.nome.startswith("unitario_"))
        ]
        return selecionados

    if modo == "completo" or cfg.versao in ("seq", "omp_diag", "auto"):
        selecionados = list(casos)
    else:
        bloco_ref = 7 if 7 in blocos else blocos[0]
        thread_ref = 4 if 4 in threads else threads[-1]
        proc_ref = 2 if 2 in procs else procs[-1]

        usar_completo = False
        if cfg.tam_bloco is None:
            usar_completo = True
        elif cfg.versao in ("omp_blocos", "omp_tarefas"):
            usar_completo = (cfg.num_threads == thread_ref) or (cfg.tam_bloco == bloco_ref)
        elif cfg.versao == "mpi_blocos":
            usar_completo = (cfg.num_procs == proc_ref) or (cfg.tam_bloco == bloco_ref)
        elif cfg.versao == "hibrido":
            usar_completo = (
                (cfg.num_procs == 2 and cfg.num_threads == 2)
                or (cfg.tam_bloco == bloco_ref)
            )

        selecionados = list(casos) if usar_completo else [caso for caso in casos if caso.essencial]

    # Evita explosao de OpenMP tasks em omp_tarefas com -b 1 nas matrizes grandes
    if cfg.versao == "omp_tarefas" and cfg.tam_bloco == 1:
        selecionados = [
            caso for caso in selecionados if max(dimensoes_caso(caso)) <= 31
        ]

    return selecionados


def verificar_binarios() -> bool:
    """Confirma que todos os binarios necessarios existem e sao executaveis."""
    faltando = [
        str(b) for b in (BIN_PARALELO, BIN_PARALELO_DEBUG, BIN_REF, BIN_REF_DEBUG) if not b.is_file()
    ]
    if faltando:
        print("ERRO: Binarios ausentes:", ", ".join(faltando), file=sys.stderr)
        print("Execute 'make all debug seq_original' em trabalho_lcs/.", file=sys.stderr)
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suite de testes automatizados para o LCS paralelo (OpenMP + MPI)."
    )
    parser.add_argument(
        "--versoes",
        nargs="+",
        default=["seq"],
        choices=list(VERSOES_CONHECIDAS),
        help="Versoes (-v) a serem testadas (padrao: seq).",
    )
    parser.add_argument(
        "--blocos",
        nargs="+",
        type=int,
        default=[1, 3, 7, 64, 256],
        help="Tamanhos de bloco (-b) para versoes blocadas (padrao: 1 3 7 64 256).",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 7, 8],
        help="Contagens de threads OpenMP (padrao: 1 2 3 4 5 7 8).",
    )
    parser.add_argument(
        "--procs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Contagens de processos MPI (padrao: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--repeticoes",
        type=int,
        default=1,
        help="Numero de repeticoes por caso/configuracao (padrao: 1).",
    )
    parser.add_argument(
        "--oversubscribe",
        action="store_true",
        default=True,
        help="Passa --oversubscribe para mpirun quando usar MPI (padrao: habilitado).",
    )
    parser.add_argument(
        "--modo",
        choices=["enxuto", "completo"],
        default="enxuto",
        help=(
            "Modo de cobertura para combinacoes de blocos/threads/processos: "
            "'enxuto' (padrao) executa todos os casos nos cortes principais e os "
            "edge cases essenciais nas demais combinacoes; 'completo' executa todos "
            "os casos em todas as combinacoes."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Numero de casos executados em paralelo na suite (padrao: 12).",
    )
    parser.add_argument(
        "--casos",
        nargs="+",
        default=None,
        help="Lista opcional de nomes de casos para filtrar a execucao (ex.: checagem de fumaca).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not verificar_binarios():
        return 1

    casos = construir_casos_teste()
    if args.casos is not None:
        nomes_filtro = set(args.casos)
        casos = [c for c in casos if c.nome in nomes_filtro]
        if not casos:
            print("ERRO: Nenhum caso de teste corresponde a --casos informado.", file=sys.stderr)
            return 1
    referencias = gerar_referencias(casos)

    total_execucoes = 0
    total_passou = 0
    resumo_por_versao: Dict[str, Tuple[int, int]] = {}
    falhas: List[Tuple[str, str, str]] = []
    workers = max(1, args.workers)

    for versao in args.versoes:
        configs = gerar_configuracoes(versao, args.blocos, args.threads, args.procs)
        if versao == "mpi_blocos":
            workers_versao = min(workers, 6)
        elif versao in ("hibrido", "auto"):
            workers_versao = min(workers, 4)
        else:
            workers_versao = workers
        passou_versao = 0
        falhou_versao = 0
        print(
            f"=== Testando versao '{versao}' ({len(configs)} configuracao(oes), "
            f"modo={args.modo}, workers={workers_versao}) ==="
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers_versao) as executor:
            for cfg in configs:
                casos_cfg = selecionar_casos_para_config(
                    casos, cfg, args.blocos, args.threads, args.procs, args.modo
                )
                futuros = [
                    (
                        caso,
                        executor.submit(
                            validar_caso,
                            caso,
                            referencias[caso.nome],
                            cfg,
                            args.repeticoes,
                            args.oversubscribe,
                        ),
                    )
                    for caso in casos_cfg
                ]
                for caso, futuro in futuros:
                    total_execucoes += 1
                    erro = futuro.result()
                    if erro is None:
                        total_passou += 1
                        passou_versao += 1
                        print(f"[PASSOU] ({cfg.descricao()}) {caso.nome}")
                    else:
                        falhou_versao += 1
                        falhas.append((cfg.descricao(), caso.nome, erro))
                        print(f"[FALHOU] ({cfg.descricao()}) {caso.nome} -> {erro}")
        resumo_por_versao[versao] = (passou_versao, falhou_versao)

    print("\n=== Resumo por Versao ===")
    for versao, (passou_v, falhou_v) in resumo_por_versao.items():
        print(f"- {versao}: executado={passou_v + falhou_v} | passou={passou_v} | falhou={falhou_v}")

    print("\n=== Resumo Geral ===")
    print(f"Total executado: {total_execucoes} | Passou: {total_passou} | Falhou: {len(falhas)}")
    if falhas:
        print("\nDetalhes das falhas:")
        for desc_cfg, nome_caso, erro in falhas:
            print(f"- ({desc_cfg}) {nome_caso}: {erro}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
