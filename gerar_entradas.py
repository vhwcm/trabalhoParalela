#!/usr/bin/env python3
"""Gerador de sequencias alfanumericas para os testes e benchmarks de LCS."""

import argparse
import random
import string


ALFABETO = string.ascii_letters + string.digits
ALFABETO_A = string.ascii_lowercase + "01234"
ALFABETO_B = string.ascii_uppercase + "56789"


def gerar_sequencia(tamanho, alfabeto, gerador):
    """Gera uma string de tamanho especificado usando o alfabeto dado."""
    return "".join(gerador.choice(alfabeto) for _ in range(tamanho))


def salvar_arquivo(caminho, conteudo):
    """Grava a sequencia no arquivo especificado."""
    with open(caminho, "w", encoding="ascii") as arquivo:
        arquivo.write(conteudo)


def main():
    parser = argparse.ArgumentParser(
        description="Gera fileA.in e fileB.in para experimentos de LCS."
    )
    parser.add_argument("tam_a", type=int, help="Tamanho da sequencia A")
    parser.add_argument("tam_b", type=int, help="Tamanho da sequencia B")
    parser.add_argument(
        "modo",
        nargs="?",
        default="aleatorio",
        choices=["aleatorio", "iguais", "diferentes"],
        help="Modo de geracao (padrao: aleatorio)",
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=42,
        help="Semente fixa para reprodutibilidade (padrao: 42)",
    )

    args = parser.parse_args()

    if args.tam_a < 0 or args.tam_b < 0:
        raise SystemExit("Erro: tam_a e tam_b devem ser nao negativos.")

    gerador = random.Random(args.semente)

    if args.modo == "aleatorio":
        seq_a = gerar_sequencia(args.tam_a, ALFABETO, gerador)
        seq_b = gerar_sequencia(args.tam_b, ALFABETO, gerador)
    elif args.modo == "iguais":
        seq_a = gerar_sequencia(args.tam_a, ALFABETO, gerador)
        if args.tam_a == args.tam_b:
            seq_b = seq_a
        elif args.tam_a == 0:
            seq_b = gerar_sequencia(args.tam_b, ALFABETO, gerador)
        else:
            repeticoes = (args.tam_b // args.tam_a) + 1
            seq_b = (seq_a * repeticoes)[: args.tam_b]
    else:  # diferentes
        seq_a = gerar_sequencia(args.tam_a, ALFABETO_A, gerador)
        seq_b = gerar_sequencia(args.tam_b, ALFABETO_B, gerador)

    salvar_arquivo("fileA.in", seq_a)
    salvar_arquivo("fileB.in", seq_b)


if __name__ == "__main__":
    main()
