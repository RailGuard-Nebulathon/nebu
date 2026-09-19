"""Official ACV linear rank-decay score."""


def rank_decay_score(ranking: list[str], truth: str) -> float:
    try:
        rank = ranking.index(truth) + 1
    except ValueError:
        return 0.0
    return (len(ranking) - (rank - 1)) / len(ranking) if ranking else 0.0

