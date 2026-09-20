"""Run the scraper, embedding model, and optimizer once."""

from __future__ import annotations

import logging

from .embeddings import Word2VecEmbedding
from .optimizer import solve_with_cost
from .scraper import scrape_connections


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    puzzle = scrape_connections()
    model = Word2VecEmbedding()
    distances = model.distance_matrix(puzzle["words"])
    groups, cost = solve_with_cost(puzzle["words"], distances)

    print(f"Optimizer total cost: {cost:.6f}\n")
    print("OPTIMIZER GROUPS                         NYT ANSWERS")
    for index in range(4):
        found = ", ".join(groups[index])
        answer = puzzle["answer"][index]
        expected = f'{answer["category"]}: {", ".join(answer["words"])}'
        print(f"{found:<40} {expected}")


if __name__ == "__main__":
    main()
