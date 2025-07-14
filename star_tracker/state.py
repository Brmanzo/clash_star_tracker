# star_tracker/state.py
import FreeSimpleGUI as sg, json, numpy as np, shutil, sys
from pathlib import Path
from typing import List, Optional

from star_tracker.presets import processingPresets, gameRulePresets, dataColumn, imageMeasurements
from star_tracker.player_utils import playerData, attackData

class currentState:
    """Holds the current state of the application, including settings, data structures, and iterators."""
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

    VENV_PYTHON = sys.executable
    ENTRY_POINT = PROJECT_ROOT / "star_tracker" / "__main__.py"
    BATCH_FILE = PROJECT_ROOT / "Program_Files" / "Star_Tracker.bat"
    ICO_FILE = PROJECT_ROOT / "Program_Files" / "star_tracker.ico"
    SHORTCUT_NAME  = "Star Tracker" 

    SETTINGS_FILE = PROJECT_ROOT / "Program_Files" / "past_files.json"
    ADVANCED_SETTINGS_FILE = PROJECT_ROOT / "Program_Files" / "advanced_settings.json"
    GAME_RULES_FILE = PROJECT_ROOT / "Program_Files" / "gamerules.json"
    HISTORY_FILE = PROJECT_ROOT / "player_history.csv"
    MEASUREMENT_FILE = PROJECT_ROOT / "Program_Files" / "measurements.json"

    if not SETTINGS_FILE.exists():
        # Create default settings file if it doesn't exist
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({}, f, indent=4)

    def __init__(self):
        """Initialize the current state with default values."""
        # GUI elements
        self.window: sg.Window|None = None
        self.settings: dict = {}
        self.advancedSettings: dict = {}
        self.gameRules: dict = {}
        self.gameRulePresets = gameRulePresets()
        self.presets = processingPresets()
        self.measurements: dict = {}
        self.measurementPresets: Optional[imageMeasurements] = None
        
        # Data structures
        self.players = []
        self.multiAccounters: dict[str, list[str]] | None = None
        self.aliasMap       : dict[str, str] | None = None
        self.seenAliases:    dict[str, set[str]]     = {}
        self.multiNextIdx   : dict[str, int] = {}

        self.enemies = []
        self.playersSeen = set()
        self.enemiesSeen = set()
        self.enemiesRanks = {}
        self.war_players:List[Optional[playerData]] = [None] * self.MAX_WAR_PLAYERS
        self.war_enemies: list[Optional[str]] = [None]*(self.MAX_WAR_PLAYERS+1)
        self.new_scores: dict[str, int] = {}
        self.editable_lines: list[str] = []

        # Measurements
        self.srcDimensions: tuple[int, int] | None = None
        self.menuTopMargin: int | None = None
        self.menuBottomMargin: int | None = None
        self.menuLeftMargin: int | None = None
        self.menuRightMargin: int | None = None
        self.menuDimensions: tuple[int, int] | None = None

        self.headerEnd: int | None = None
        self.lineBegin: int | None = None
        self.lineEnd: int | None = None
        self.attackLinesDimensions: tuple[int, int] | None = None

        self.rankEnd: int | None = None
        self.levelEnd: int | None = None
        self.playerEnd: int | None = None
        self.enemyStart: int | None = None
        self.starsColEnd: int | None = None
        self.enemyEnd: int | None = None
        self.percentageBegin: int | None = None
        self.firstStar: int | None = None
        self.starsBegin: int | None = None
        self.percentageEnd: int | None = None
        self.realStarsEnd: int | None = None

        # Column data
        self.rankCol: dataColumn|None = None
        self.levelCol: dataColumn|None = None
        self.playerCol: dataColumn|None = None
        self.enemyCol: dataColumn|None = None
        self.percentageCol: dataColumn|None = None
        self.starsCol: dataColumn|None = None

        # Path
        self.file_list: List[Path]|None = None
        self.image_path: Path|None = None
        self.debug_name: List[str]|None = None

        # Images
        self.src: np.ndarray|None = None
        self.srcL: np.ndarray|None = None
        self.menu: np.ndarray|None = None
        self.menuL: np.ndarray|None = None
        self.attackLines: np.ndarray|None = None
        self.attackLinesL: np.ndarray|None = None

        # Iterators
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

def print_to_gui(s: currentState, text_to_print: str):
    """A helper function to print text to the Multiline element."""
    # The '+=' appends the new text, and '\n' adds a newline.
    if s.window is None or '-OUTPUT-' not in s.window.AllKeysDict:
        return
    output_elem = s.window['-OUTPUT-'] if s.window is not None and '-OUTPUT-' in s.window.AllKeysDict else None
    if output_elem is not None:
        output_elem.update(value=text_to_print + '\n', append=True)
        s.window.refresh() # Force the GUI to update
