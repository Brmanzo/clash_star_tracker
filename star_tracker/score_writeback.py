# star_tracker/score_writeback.py
import csv
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Dict

def load_history(path) -> Tuple[list[str], OrderedDict]:
    '''Load csv file of previous war data'''
    table = OrderedDict()
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.reader(f, skipinitialspace=True)
        header = next(rdr, None)
        for row in rdr:
            if not row:
                continue
            player = row[0].strip()
            scores = [c.strip() for c in row[1:-1]]
            table[player] = scores
    return header, table   

def merge_new_war(table, new_scores):
    '''Calculate new column and total score column'''
    prev_cols = len(next(iter(table.values()), []))
    # If not present in war, indicate with underscore
    for row in table.values():
        row.append("_")

    for raw_name, stars in new_scores.items():
        player = raw_name.strip()
        if player in table:
            table[player][-1] = str(stars)
        else:
            table[player] = ["_"] * prev_cols + [str(stars)]

def rebuild_totals(table) -> Dict[str, int]:
    '''Append new war data and new sum to the appropriate players within csv'''
    tot_dict = {}
    for player, row in table.items():
        tot = sum(int(x) for x in row if x.isdigit())
        tot_dict[player] = tot
    return tot_dict

def write_history(path, table, totals) -> None:
    '''Writes modified csv back to file'''
    n_wars = len(next(iter(table.values())))
    header = ["Player"] + [f"War-{i+1}" for i in range(n_wars)] + ["Total"]

    ordered = sorted(
        table.items(),
        key=lambda kv: (-totals[kv[0]], kv[0])
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for player, row in ordered:
            wr.writerow([player] + row + [totals[player]])
    print("Written to", path)

def print_leaderboard(table, totals, width_name=22) -> None:
    '''Print "Rank  Name  Total" to the terminal.'''
    ordered = sorted(
        table.items(),
        key=lambda kv: (-totals[kv[0]], kv[0])     # same sort as CSV
    )

    print("\n=== Current Leaderboard ===")
    for i, (player, _) in enumerate(ordered, start=1):
        # discord_name = display_name(player)
        print(f"{i:>2}. {player.ljust(width_name)} {totals[player]}")

def load_player_list(path: str | Path) -> list[str]:
    """Read one name per line, ignore blank lines and trim whitespace."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"player file not found: {p}")

    with p.open(encoding="utf-8") as f:
        names = [line.strip()                       # remove \n and spaces
                 for line in f
                 if line.strip()]                   # drop empty lines
    # optional: make them unique while preserving order
    seen, unique = set(), []
    for n in names:
        if n not in seen:
            unique.append(n)
            seen.add(n)
    return unique