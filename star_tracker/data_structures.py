# star_tracker/data_structures.py
import json, os, shutil, sys
from pathlib import Path
from typing import List, Optional
from .score_writeback import load_player_list
import numpy as np

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
    
    def __init__(self, rank: int|None, name: str, attacks: List[attackData]):
        self.rank = rank
        self.name = name
        self.attacks = attacks

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
                if attack_diff <= -5 and '_' in attack.score:
                    total_score -= 1
                # If dropping more than 10 and not cleaning, should earn no points
                if attack_diff <= -10 and '★' not in attack.score:
                    total_score -= attack.score.count('☆')
                # If attacking up 5 or more ranks, and scoring a new star, earn an extra point
                if attack_diff >= 5 and '☆' in attack.score:
                    total_score += 1
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
    

class currentState:
    MAX_WAR_PLAYERS = 50
    HOME = Path.home()
    PLAYERS_FILE = HOME / "Desktop" / "Clash" / "OperatingData" / "players.txt"

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    IMAGES_DIR   = PROJECT_ROOT / "Images"
    OUT_DIR      = PROJECT_ROOT / "Debug"

    OUT_DIR.mkdir(exist_ok=True)
    IMG_EXTS     = (".png", ".jpg", ".jpeg")

    TESS_EXE      = shutil.which("tesseract")
    MODEL_NAME    = "eng"
    # PSM 7 for reading lines of text
    PLAYER_CONFIG = f"--psm 7 -l {MODEL_NAME}"
    # PSM 10 for single character/num recognition
    RANK_CONFIG   = f"--psm 10 -l {MODEL_NAME} -c tessedit_char_whitelist=0123456789lLiIoOsSzZ|"

    SETTINGS_FILE = PROJECT_ROOT / "clash_star_tracker_settings.json"
    if not SETTINGS_FILE.exists():
        # Create default settings file if it doesn't exist
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({}, f, indent=4)

    def __init__(self):
        """Initialize the current state with default values."""
        self.players = []
        self.multiAccounters = None
        self.enemies = []
        self.playersSeen = set()
        self.enemiesSeen = set()
        self.enemiesRanks = {}
        self.war_players:List[Optional[playerData]] = [None] * self.MAX_WAR_PLAYERS
        self.new_scores: dict[str, int] = {}

        self.rankCol: dataColumn|None = None
        self.levelCol: dataColumn|None = None
        self.playerCol: dataColumn|None = None
        self.enemyCol: dataColumn|None = None
        self.percentageCol: dataColumn|None = None
        self.starsCol: dataColumn|None = None

        self.file_list: List[Path]|None = None
        self.image_path: Path|None = None
        self.debug_name: List[str]|None = None

        self.src: np.ndarray|None = None
        self.srcL: np.ndarray|None = None
        self.attackLines: np.ndarray|None = None
        self.attackLinesL: np.ndarray|None = None

        self.verbose = True if "--v" in sys.argv else False

        self.abs_pos = 0
        self.lineTop = 0
        self.lineBottom = 0
        self.nextLineTop = 0
        self.lineHeight = 0
        self.linesHeight = 0

        self.fileNum = 1
        self.lineNum = 0

    def reset(self) -> None:
        """Reset the current state for a new image processing run."""
        self.src = None
        self.srcL = None
        self.attackLines = None
        self.attackLinesL = None

        self.rankCol = None
        self.levelCol = None
        self.playerCol = None
        self.enemyCol = None
        self.percentageCol = None
        self.starsCol = None

        self.lineTop = 0
        self.lineBottom = 0
        self.nextLineTop = 0
        self.lineHeight = 0
        self.linesHeight = 0

        self.fileNum += 1
        self.lineNum = 0