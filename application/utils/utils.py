# Define multipliers
TOP_5_MULTIPLIER = 1.25
TOP_3_MULTIPLIER = 1.5
VICTORY_MULTIPLIER = 2.0


def calc_sg_score(kill_count, ranking):
    """
    Calculate the sgScore based on the ranking and kill count.
    - Victory (Rank 1) → x2.0
    - Top 3 (Ranks 2-3) → x1.5
    - Top 5 (Ranks 4-5) → x1.25
    - Otherwise → No multiplier
    """
    # Ensure ranking and kill count are valid numbers
    ranking = float(ranking)
    kill_count = float(kill_count)

    print(f"Ranking: {ranking}, Kill Count: {kill_count}")  # Debug: print the values

    if ranking == 1:
        return kill_count * VICTORY_MULTIPLIER
    elif 2 <= ranking <= 3:  # Equivalent to ranking in [2, 3]
        return kill_count * TOP_3_MULTIPLIER
    elif 4 <= ranking <= 5:  # Equivalent to ranking in [4, 5]
        return kill_count * TOP_5_MULTIPLIER
    else:
        return kill_count  # No multiplier for ranks greater than 5
