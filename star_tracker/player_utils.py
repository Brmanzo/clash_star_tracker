# # File: star_tracker/player_utils.py
from __future__ import annotations
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from star_tracker.state import currentState  

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
        if self.rank is None or self.target is None or self.score is None:
            return "No Attack, ___, 0"
        return ", ".join([str(self.rank), self.target, self.score])

    def __init__(self, rank: int|None, target: str, score: str):
        self.rank:int|None     = rank
        self.target:str|None   = target
        self.score:str|None    = score

class playerData:
    """Data container for a single player in the war."""
    
    def __init__(self, s:"currentState", rank: int|None, name: str, attacks: List[attackData]):
        self.rank = rank
        self.name = name
        self.attacks = attacks
        self.presets = s.gameRulePresets

    def total_score(self) -> int:
        """Calculates the final score based on game rules."""
        # Initialize score to 0 before the loop
        if self.rank is None or not self.attacks:
            return 0
        total_score = 0
        for attack in self.attacks:
            if not hasattr(attack, 'score') or not hasattr(attack, 'rank'):
                continue # Skip if attack object is not valid
            if attack.score is not None and attack.rank is not None:    
                total_score += attack.score.count("★") + attack.score.count("☆")

                # If dropping more than 5 ranks, and not a 3 star, lose a point
                attack_diff = self.rank - int(attack.rank)
                if attack_diff <= self.presets.noThreeStarDroppingThreshold and '_' in attack.score:
                    if self.presets.noThreeStarDroppingPenalty == "Negate earned stars":
                        total_score -= (attack.score.count("★") + attack.score.count("☆"))
                    else:
                        total_score += int(self.presets.noThreeStarDroppingPenalty)
                # If dropping more than 10 and not cleaning, should earn no points
                if attack_diff <= self.presets.droppingForFirstAttackThreshold and '★' not in attack.score:
                    if self.presets.droppingForFirstAttackPenalty == "Negate earned stars":
                        total_score -= (attack.score.count("★") + attack.score.count("☆"))
                    else:
                        total_score += int(self.presets.droppingForFirstAttackPenalty)
                # If attacking up 5 or more ranks, and scoring a new star, earn an extra point
                if attack_diff >= self.presets.successfulJumpThreshold and '☆' in attack.score:
                    total_score += self.presets.successfulJumpBonus
            # Handles cases where attack.rank might not be a valid number
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