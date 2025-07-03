# star_tracker/data_structures.py
from typing import List 

class dataColumn:
    '''Records the absolute position of the column in the original image
    
    Given the relative end point, constructs an object reporting the beginning as
    the previous column's end as well as the resulting width of the column'''
    abs_pos = 0

    def __init__(self, end, begin=0):
        self.begin = dataColumn.abs_pos + begin
        self.end   = end + self.begin
        self.width = self.end - self.begin

        dataColumn.abs_pos += self.width + begin

class attackData:
    '''Data container for each attack in war'''
    def tabulate_attack(self) -> str:
        """Returns a single, formatted CSV string for the attack."""
        return ", ".join([str(self.rank), self.target, self.score])

    def __init__(self, rank: int, target: str, score: str):
        self.rank     = rank
        self.target   = target
        self.score    = score

class playerData:
    """Data container for a single player in the war."""
    
    def __init__(self, rank: int, name: str, attacks: List[attackData]):
        self.rank = rank
        self.name = name
        self.attacks = attacks

    def total_score(self) -> int:
        """Calculates the final score based on game rules."""
        # Initialize score to 0 before the loop
        total_score = 0
        for attack in self.attacks:
            if not hasattr(attack, 'score') or not hasattr(attack, 'rank'):
                continue # Skip if attack object is not valid

            total_score += attack.score.count("★") + attack.score.count("☆")

            try:
                # If dropping more than 5 ranks, and not a 3 star, lose a point
                attack_diff = self.rank - int(attack.rank)
                if attack_diff <= -5 and '_' in attack.score:
                    total_score -= 1
                # If dropping more than 10 and not cleaning, should earn no points
                if attack_diff <= -10 and '☆' not in attack.score:
                    total_score -= attack.score.count('★')
                # If attacking up 5 or more ranks, and scoring a new star, earn an extra point
                if attack_diff >= 5 and '★' in attack.score:
                    total_score += 1
            # Handles cases where attack.rank might not be a valid number
            except (ValueError, TypeError):
                pass
        return total_score

    def tabulate_player(self) -> str:
        """Returns a single, formatted CSV string for the player."""
        
        attack1_str = self.attacks[0].tabulate_attack() if len(self.attacks) > 0 else "No Attack 1, ___, 0"
        attack2_str = self.attacks[1].tabulate_attack() if len(self.attacks) > 1 else "No Attack 2, ___, 0"

        clean_attack1 = attack1_str.replace('\n', ' ').strip()
        clean_attack2 = attack2_str.replace('\n', ' ').strip()

        parts = [
            str(self.rank).replace('\n', ' ').strip(),
            self.name.replace('\n', ' ').strip(),
            clean_attack1,
            clean_attack2,
            str(self.total_score())
        ]
        return ", ".join(parts)