#!/usr/bin/env python3
# main.py
# ------------------------------------------------------------
import cv2, json, os, pathlib, pytesseract, shutil,  sys
import numpy as np

from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from .data_structures import dataColumn, attackData, playerData
from .score_writeback import load_player_list, load_history, merge_new_war, rebuild_totals, write_history, print_leaderboard
from .preprocessing import measure_image, sample_image, debug_oscilloscope, debug_image, count_peaks
from .ocr import auto_correct_num, auto_correct_player, preprocess_line, score_from_stars
# ------------------------------------------------------------

# Set up paths
HOME = Path.home()

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

# Set the path to the Tesseract executable
if not TESS_EXE:
    sys.exit("Tesseract executable not found. Please install Tesseract and ensure it is in your PATH.")
rootdir = os.path.dirname(os.path.realpath(__file__))
pytesseract.pytesseract.tesseract_cmd = TESS_EXE        

# Load player list from file
PLAYERS_FILE = HOME / "Desktop" / "Clash" / "OperatingData" / "players.txt"
if PLAYERS_FILE.is_file():
    players = load_player_list(PLAYERS_FILE)
else:
    print(f"Warning: player file not found: {PLAYERS_FILE}, using empty player list.")
    players = []
    
enemies = []
enemiesRanks = {}
playersSeen = set()
enemiesSeen = set()
# Open Multi Accounters from JSON file for aliasing names
with open("multi_accounts.json", encoding="utf-8") as f:
    multiAccounters = json.load(f)

# verbose mode
verbose = True if "--v" in sys.argv else False
# debug mode
debug = True if "--db" in sys.argv else False

MAX_WAR_PLAYERS  = 50
MENU_BG_TH       = 0.3
BLACK_TH         = 0.01
WHITE_TH         = 0.99

PLAYER_NAME_TOL  = 100
PX_MARGIN        = 10
# ------------------------------------------------------------

def main() -> None:
    # Data structure to log current war data from screenshots
    war_players:List[Optional[playerData]] = [None] * MAX_WAR_PLAYERS

    if not IMAGES_DIR.is_dir():
        sys.exit(f"Folder not found: {IMAGES_DIR}")

    for dirpath, _, filenames in os.walk(IMAGES_DIR):
        k = 0
        for fname in filenames:
            # Skip non-image files
            if not fname.lower().endswith(IMG_EXTS):
                continue
            dataColumn.abs_pos,  nextLineTop, abs_pos = 0, 0, 0
            k += 1
            if verbose: print(f"Processing {k} of {len([f for f in filenames if f.lower().endswith(IMG_EXTS)])}: {fname}")

            # Construct the full path to the image file
            image_path = pathlib.Path(dirpath) / fname
            debug_name = pathlib.Path("_".join(fname.split(" "))).stem + "_proc.png"
            
            src = cv2.imread(image_path)

            if src is None:
                sys.exit(f"cannot read: {image_path}, skipping")

            # -------------------------------------------------------------- Center screenshot on menu lines --------------------------------------------------------------
            srcL = cv2.cvtColor(src, cv2.COLOR_BGR2HLS)[:, :, 1]

            # Scan image by row and column to find menu margins from war background (based on lightness)
            menuTopMargin, menuBottomMargin = measure_image(srcL, MENU_BG_TH, behavior="relative threshold, average, by row, first rise, last, fall")
            menuLeftMargin, menuRightMargin = measure_image(srcL, MENU_BG_TH, behavior="relative threshold, average, by col, first rise, last, fall")

            menu = srcL[menuTopMargin : menuBottomMargin, menuLeftMargin : menuRightMargin]
            if debug:
                debug_oscilloscope(cv2.rotate(menu.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE), f"{debug_name}_{k}_y_menu_axis", None, OUT_DIR)
                debug_oscilloscope(menu.copy(), f"{debug_name}_{k}_x_menu_axis", None, OUT_DIR)

            # ------------------------------------------------------------ Crop Screenshot to only the attack lines ------------------------------------------------------------

            col_menu_max_avg_TH = sample_image(menu, "max, absolute, average, by col", None, eps=0.001)*0.99

            row_menu_min_TH = sample_image(menu, "max, absolute, minimum, by row", None, eps=0.001) * 0.97

            # Scan from top, past the headers to get to the top of the first line, leave the whitespace following the last line
            headerEnd = measure_image(menu[PX_MARGIN:,:], row_menu_min_TH, behavior="absolute threshold, minimum, by row, first fall, next, fall")[1]
            if debug: print(f"header end: {headerEnd}, row_menu_min_TH: {row_menu_min_TH}")

            # Scan from edge of menu to lines, by targetting when average drops below max average
            lineBegin, lineEnd = measure_image(menu[headerEnd:, :], col_menu_max_avg_TH, behavior=f"absolute threshold, average, by col, first fall, last, rise")
            if debug: print(f"threshold: {col_menu_max_avg_TH}, line begin: {lineBegin}, line end: {lineEnd} ")
            
            # Package the menu as a single lightness channel with the correct dimensions of menu
            attackLines  = menu[headerEnd:, lineBegin:lineEnd]
            menuHeight, _ = attackLines.shape[:2]
            if debug: debug_image(debug_name, 0, attackLines, "attackLines", OUT_DIR)

            # --------------------------------------------------------------- Determine data locations from lines ---------------------------------------------------------------
            # Sample image properties - outliers.
            if debug: debug_oscilloscope(attackLines[:, 15:-15].copy(), f"{debug_name}_{k}_x_attack_lines_axis", None, OUT_DIR)

            # Adaptive thresholding counts the unique jumps in d/dx (avg) which demarcate the explicit columns
            AL_global_max_min = sample_image(attackLines[:, 15:-15], "max, absolute, minimum, by col", None, eps=0.001)*0.99
            if debug: print(f"AL_global_max_min: {AL_global_max_min}")
            line_data_sep_TH = sample_image(attackLines[:, 15:-15], "max, relative, average, by col", None, eps=0.0005)*0.99
            if debug: print(f"line_data_sep_TH: {line_data_sep_TH}")

            # Rank begins at 0 and ends at first explicit column
            _, rankEnd  = measure_image(attackLines, line_data_sep_TH, behavior="relative threshold, average, by col, first fall, next, rise")
            rankCol = dataColumn(rankEnd)

            if debug: print(f"rankEnd: {rankEnd}, line_data_sep_TH: {line_data_sep_TH}")
            # Level ends at the second explicit column
            _, levelEndAvg = measure_image(attackLines[:, rankCol.end:], BLACK_TH, behavior="absolute threshold, minimum, by col, first fall, next, fall")
            levelCol = dataColumn(levelEndAvg)

            # When player names bleed over the explicit level end column, safer to use presence of black to determine player beginning
            if debug: print(f"levelCol.end: {levelCol.end}")
            # Player ends at the third explicit column
            _, playerEnd = measure_image(attackLines[:, levelCol.end + PLAYER_NAME_TOL:], line_data_sep_TH, behavior="relative threshold, average, by col, from start, next, fall")
            playerCol = dataColumn(playerEnd + PLAYER_NAME_TOL)

            if debug: print(f"playerCol.end: {playerCol.end}")

            # Enemy begins when first appearance of black in fourth explicit column
            # When minimum lightness is less than 0.01, black is present
            _, enemyStart = measure_image(attackLines[:, playerCol.end:], BLACK_TH, behavior="absolute threshold, minimum, by col, from start, next, rise")

            if debug: print(f"enemyStart: {playerCol.end + enemyStart}")

            # Scans ahead to find the final explicit column where attack ends and score begins
            _, starsColEnd = measure_image(attackLines[:, playerCol.end + PX_MARGIN:], line_data_sep_TH, behavior=f"relative threshold, average, by col, from start, next, rise while min > {AL_global_max_min*0.95}")
            starsColEnd = starsColEnd + PX_MARGIN + dataColumn.abs_pos

            if debug: print(f"starsColEnd: {starsColEnd}")
            # adaptive thresholding for local minimum, removing global maximum from attack column
            AL_local_max_TH = sample_image(attackLines[:, enemyStart + PX_MARGIN:starsColEnd - PX_MARGIN], "max, absolute, minimum, by col", AL_global_max_min, eps=0.01) * 0.95
            
            if debug: print(f"AL_local_max_TH: {AL_local_max_TH}")
  
            # Enemy ends when minimum lightness returns to local maximum, skip ahead 50 in case longest enemy rank spacing results in false max
            _, enemyEnd = measure_image(attackLines[:, playerCol.end + enemyStart + PLAYER_NAME_TOL:], AL_local_max_TH, behavior=f"absolute threshold, minimum, by col, from start, next, rise")
            enemyCol = dataColumn(enemyEnd + PX_MARGIN + PLAYER_NAME_TOL, enemyStart - PX_MARGIN)
            enemyCol.begin -= PX_MARGIN

            if debug: print(f"enemyEnd: {playerCol.end + enemyStart + enemyEnd + PLAYER_NAME_TOL}")
            if debug: print(f"enemyCol.end: {playerCol.end + enemyStart + enemyEnd + PLAYER_NAME_TOL}")

            _, percentageBegin = measure_image(attackLines[:, enemyCol.end:], AL_local_max_TH, behavior=f"absolute threshold, minimum, by col, from start, next, fall")
            # Center the end of enemy column in between the beginning of percentage
            enemyCenter = (percentageBegin//2) + 1
            enemyCol.end += enemyCenter
            dataColumn.abs_pos += enemyCenter
            percentageBegin -= (percentageBegin//2)
            percentageBegin += enemyCol.end

            # First star occurs with presence of white, scan ahead to the first star
            _, firstStar = measure_image(attackLines[:, percentageBegin:], WHITE_TH, behavior="absolute threshold, maximum, by col, from start, next, rise")
            firstStar += percentageBegin

            if debug: print(f"percentage begin: {percentageBegin}, first star: {firstStar}")

            # Scan backwards from first star for the first drop in local minimum indicating end of percentage
            starsBegin, percentageEnd = measure_image(cv2.flip(attackLines[:, percentageBegin:firstStar], 1), AL_local_max_TH ,behavior=f"absolute threshold, minimum, by col, first rise, next, fall")
            percentageEnd = firstStar - percentageEnd
            starsBegin = firstStar - starsBegin
            percentageCol = dataColumn(starsBegin - percentageBegin + enemyCenter)

            if debug: print(f"percentageCol.end: {percentageCol.end}, starsColEnd: {starsColEnd}")

            # Scan backwards from explicit attack column end to first presence of black, indicating edge of stars
            _, realStarsEnd = measure_image(cv2.flip(attackLines[:, percentageCol.end:starsColEnd - PX_MARGIN], 1), AL_local_max_TH ,behavior=f"absolute threshold, minimum, by col, from start, next, fall")
            realStarsEnd = starsColEnd - PX_MARGIN - realStarsEnd
            starWidth = realStarsEnd - percentageCol.end

            # If no new third star, realStarsEnd may be the wrong width
            # Two peaks in lightness per star, if less than 6, only a new second star in screenshot
            # if less than 3, then only a new first star in screenshot
            peaks = count_peaks(attackLines[:, percentageCol.end:starsColEnd], WHITE_TH)
            # Adjust width to true width of stars based on measured width
            if peaks >= 4 and peaks < 6:
                starWidth = starWidth * (3/2)
            elif peaks < 3:
                starWidth = starWidth * 3
            starsCol = dataColumn(starWidth)
            
            cropped = src[menuTopMargin + headerEnd : menuBottomMargin, menuLeftMargin + lineBegin : menuLeftMargin + lineEnd]
            croppedL = cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS)[:, :, 1]

            linesHeight, _ = croppedL.shape[:2]
            dbgx = croppedL.copy()
            dbgy = croppedL.copy()
            if debug:
                debug_oscilloscope(dbgx, f"{debug_name}_{k}x_axis_columns", {rankCol, levelCol, playerCol, enemyCol, percentageCol, starsCol}, OUT_DIR)
                debug_oscilloscope(cv2.rotate(dbgy, cv2.ROTATE_90_COUNTERCLOCKWISE), f"{debug_name}_{k}_y_axis", None, OUT_DIR)
                debug_oscilloscope(dbgx.copy(), f"{debug_name}_{k}_x_axis", None, OUT_DIR)

            # Initializing state variables for processing each line
            name, rank = None, None
            abs_pos, lineTop, nextLineTop, lineHeight, line_idx = 0, 0, 0, 0, 0
            # ------------------------------------------------------------ Process each line in the menu ------------------------------------------------------------
            # Adaptive thresholding for white between lines
            new_line_TH = sample_image(attackLines, "max, absolute, minimum, by row", None, eps=0.01) * 0.97
            if debug: print(f"new_line_TH: {new_line_TH}")

            while True:
                if debug: print(f"nextLineTop: {nextLineTop}, lineHeight: {lineHeight}, menuHeight: {menuHeight}")
                abs_pos += nextLineTop

                lineTop = abs_pos
                lineBottom, nextLineTop = measure_image(croppedL[lineTop + PX_MARGIN:, :], new_line_TH, behavior="absolute threshold, minimum, by row, first rise, next, fall")

                if nextLineTop == 0:
                    lineBottom = linesHeight
                else:
                    lineBottom += lineTop + PX_MARGIN
                    nextLineTop += PX_MARGIN  # convert relative → absolute

                lineHeight = lineBottom - lineTop
                if debug: print(f"loop {line_idx + k}: line Top: {lineTop}, line Bottom: {lineBottom}")

                # ---------------------------------------------------- Rank Processing ----------------------------------------------------
                if debug: print(f"rankCol Begin: {rankCol.begin}, rankCol End: {rankCol.end}")
                rankCrop = cropped[lineTop:lineBottom ,rankCol.begin:rankCol.end]
                rank_preproc = preprocess_line(rankCrop, line=True)

                rank_preproc = rank_preproc[PX_MARGIN : -PX_MARGIN, PX_MARGIN:- PX_MARGIN]
                
                # Specify single character page segmentation, and restrict output to integers only
                rank = pytesseract.image_to_string(rank_preproc, config=RANK_CONFIG)

                if debug: 
                    print(f"text from player screenshot: {rank}")
                    debug_image(debug_name, line_idx + k, rank_preproc, "rank_preproc", OUT_DIR)
                    debug_image(debug_name, line_idx + k, rankCrop, "rank_crop", OUT_DIR)
                rank = auto_correct_num(rank)

                # ------------------------------------------------- Player Name Processing -------------------------------------------------
                playerCrop = cropped[lineTop:lineBottom ,playerCol.begin:playerCol.end]
                player_preproc = preprocess_line(playerCrop, line=True)

                player = auto_correct_player(pytesseract.image_to_string(player_preproc, config=PLAYER_CONFIG), enemies=enemies, players=players)

                if debug:
                    print(f"text from player screenshot: {player}")
                    debug_image(debug_name, line_idx + k, player_preproc, "player_preproc", OUT_DIR)
                    debug_image(debug_name, line_idx + k, playerCrop, "player_crop", OUT_DIR)

                # -------------------------------------------------- Attack Processing --------------------------------------------------
                def process_attack(attackImg: np.ndarray, lineTop: int, lineBottom: int, lineHeight: int, enemyCol: dataColumn, attackNum: int) -> attackData:
                    """Process a single attack line and return an attackData object."""

                    # Split top half or bottom half of the row depending on attack number
                    
                    row_slice   = attackImg[lineTop:lineBottom, :]
                    enemy_slice = row_slice[:, enemyCol.begin:enemyCol.end]
                    stars_slice = row_slice[:, starsCol.begin:starsCol.end]

                    attackLine  = np.array_split(enemy_slice,  2, axis=0)
                    scoreLine  = np.array_split(cv2.cvtColor(stars_slice, cv2.COLOR_BGR2HLS)[:, :, 1], 2, axis=0)

                    # Convert to Lightness and sample minimum to separate player rank from name
                    attackCrop = cv2.cvtColor(attackLine[attackNum - 1], cv2.COLOR_BGR2HLS)[:, :, 1]
                    text_menu_th = sample_image(attackCrop, "max, absolute, minimum, by col", None, eps=0.01) * 0.99
                    enemyRankBegin, enemyNameBegin = measure_image(attackCrop, text_menu_th, behavior="absolute threshold, minimum, by col, first fall, next, rise")
                    
                    # Preprocess original image to read cropped sections using different configurations
                    attackPreproc = preprocess_line(attackLine[attackNum - 1], line=True)
                    enemy_rank = auto_correct_num(pytesseract.image_to_string(attackPreproc[:, enemyRankBegin:enemyNameBegin], config=RANK_CONFIG))
                    enemy = auto_correct_player(pytesseract.image_to_string(attackPreproc[:, enemyNameBegin:], config=PLAYER_CONFIG), enemy=True, enemies=enemies, players=players)

                    if debug:
                        print(f"text_menu_th: {text_menu_th}, enemy_rank: {enemy_rank}, enemy: {enemy}, enemyRankBegin {enemyRankBegin}, enemyNameBegin: {enemyNameBegin}")
                        debug_oscilloscope(attackCrop, f"{debug_name}_{line_idx + k}_attack{attackNum}_separating_rank_and_name", None, OUT_DIR)
                        debug_image(debug_name, line_idx + k, attackCrop[:, enemyRankBegin:enemyNameBegin], f"attack{attackNum}_rank_crop", OUT_DIR)
                        debug_image(debug_name, line_idx + k, attackCrop[:, enemyNameBegin:], f"attack{attackNum}_name_crop", OUT_DIR)

                    if enemy is None:
                        return(attackData(None, "No attack", "___"))
                        # If we've seen this enemy before, use the stored rank
                    else:
                        if enemy_rank is None and enemy is not None:
                            if enemy in enemiesSeen:
                                enemy_rank = enemiesRanks.get(enemy, None)
                                # If we haven't seen this enemy before, assume greatest unseen rank
                            else:
                                ranks = set(enemiesRanks.values())
                                top = max(ranks) if ranks else 0
                                enemy_rank = next((n for n in range(top, 0, -1) if n not in ranks), top + 1)

                            if debug:
                                print(f"text from attack {attackNum} screenshot: {enemy_rank}, {enemy}")
                                debug_image(debug_name, line_idx + k, attackCrop, f"attack{attackNum}_crop", OUT_DIR)

                        # Scan vertically to remove white space above and below stars

                        starsTop, starsBottom = measure_image(scoreLine[attackNum - 1], 0.01, behavior="stat comparison, min < average, by row, divergence, last, convergence")
                        if debug: 
                            print(f"starsTop: {starsTop}, starsBottom: {starsBottom}")
                            debug_oscilloscope(cv2.rotate(scoreLine[attackNum - 1], cv2.ROTATE_90_COUNTERCLOCKWISE), f"{debug_name + str(line_idx + k)}_stars{attackNum}_y_axis", None, OUT_DIR)

                        splitStarsLine = scoreLine[attackNum - 1]
                        if starsTop - 5 > 0: starsTop -= 5
                        if starsBottom + 5 < splitStarsLine.shape[0]: starsBottom += 5
                        # Split the stars line into three parts, each part is a star
                        # Each part is 1/3 of the width of the stars line, with a margin of 5 pixels on each side
                        stars = np.array_split(splitStarsLine[starsTop:starsBottom, :], 3, axis=1)
                        # Score is a 3 character string of stars earned in attack
                        score = f"{score_from_stars(stars[0])}{score_from_stars(stars[1])}{score_from_stars(stars[2])}"

                        if debug:
                            debug_oscilloscope(splitStarsLine[starsTop:starsBottom, :], f"{debug_name + str(line_idx + k)}_stars{attackNum}_x_axis", None, OUT_DIR)
                            debug_image(debug_name, line_idx + k, splitStarsLine[starsTop:starsBottom, :], f"attack{attackNum}StarsFinalCrop", OUT_DIR)
                            print(score)
                        return(attackData(enemy_rank, enemy, score))
                    
                # Process both attacks in the line
                attack1 = process_attack(cropped, lineTop, lineBottom, lineHeight, enemyCol, attackNum=1)
                attack2 = process_attack(cropped, lineTop, lineBottom, lineHeight, enemyCol, attackNum=2)
                # -------------------------------------------------- Player Creation --------------------------------------------------
                
                # If multiaccount detected with identical name, append number to name
                curr_player = playerData(rank, player, [attack1, attack2])
                base_name = curr_player.name
                aliases = multiAccounters.get(base_name, [])
                # If new player, store attacks and remember in war player array
                if curr_player.name and curr_player.name not in playersSeen:
                    need_free = (
                        curr_player.rank is None
                        or curr_player.rank >= len(war_players)
                        or war_players[curr_player.rank] is not None
                    )
                    # If rank misread or already occupied insert at next available placement
                    if need_free:                      
                        j = 1
                        while war_players[j] is not None:
                            j += 1
                        curr_player.rank = j

                    if aliases and war_players[curr_player.rank] is None and curr_player.name not in playersSeen:
                        curr_player.name = aliases.pop(0)
                        if not aliases:
                            multiAccounters[base_name] = []
                        if len(multiAccounters[base_name]) == 0:
                            playersSeen.add(base_name)

                    war_players[curr_player.rank] = curr_player
                    playersSeen.add(curr_player.name)
                    if curr_player.attacks is not None:
                        for attack in curr_player.attacks:
                            if attack.target is not None and attack.target not in enemiesSeen:
                                enemiesSeen.add(attack.target)
                                enemiesRanks[attack.target] = attack.rank
                    if verbose: print(curr_player.tabulate_player())

                if lineBottom + lineHeight >= menuHeight:
                    break
                
                line_idx += 1
    # -------------------------------------------------- Calculate Final Score --------------------------------------------------
    print("\n--- Final War Data with Scores ---")
    for player in war_players:
        if player is not None:
            print(player.tabulate_player())
        else: continue
    
    new_scores = {}
    for player in war_players:
        if player is None or player.name is None:
            continue
        name  = player.name
        starsCropped = player.total_score()
        new_scores[name] = starsCropped

    csv_path = os.path.join(rootdir, "../player_history.csv")

    # Load old history (or start fresh)
    try:
        _, history = load_history(csv_path)
    # If first ever run, create empty history
    except FileNotFoundError:
        history = OrderedDict()

    # Merge this war and update totals
    merge_new_war(history, new_scores)
    totals = rebuild_totals(history)

    # Write the full file back, still sorted by Total ↓
    print_leaderboard(history, totals)
    write_history(csv_path, history, totals)

if __name__ == "__main__":
    main()
