#!/usr/bin/env python3
import argparse
import os
import random


def gerar_dna(tamanho: int, semente: int) -> str:
    rng = random.Random(semente)
    bases = "ACGT"
    return "".join(rng.choice(bases) for _ in range(tamanho))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera pares deterministicos de sequencias DNA.")
    parser.add_argument("--tamanho", "-n", type=int, required=True, help="Comprimento das sequencias.")
    parser.add_argument("--semente", "-s", type=int, default=42, help="Semente fixa para geracao.")
    parser.add_argument(
        "--saida",
        "-o",
        default="/home/felps/Documents/paralela/trabalho_lcs/experimentos/entradas",
        help="Diretorio de saida.",
    )
    args = parser.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    arq_a = os.path.join(args.saida, f"dna_{args.tamanho}_A.in")
    arq_b = os.path.join(args.saida, f"dna_{args.tamanho}_B.in")

    seq_a = gerar_dna(args.tamanho, args.semente)
    seq_b = gerar_dna(args.tamanho, args.semente + 1)

    with open(arq_a, "w", encoding="ascii") as fa:
        fa.write(seq_a + "\n")
    with open(arq_b, "w", encoding="ascii") as fb:
        fb.write(seq_b + "\n")

    print(f"Entradas geradas: {arq_a} ({len(seq_a)} chars) e {arq_b} ({len(seq_b)} chars), semente={args.semente}")


if __name__ == "__main__":
    main()
