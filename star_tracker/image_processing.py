import numpy as np
import cv2
import pytesseract
import sys
from .data_structures import attackData, playerData, currentState
from .preprocessing import sample_image, measure_image, debug_image, debug_oscilloscope
from .ocr import auto_correct_num, auto_correct_player, score_from_stars, preprocess_line

PX_MARGIN = 10
STAR_MARGIN = 5
MODEL_NAME    = "eng"
# PSM 7 for reading lines of text
PLAYER_CONFIG = f"--psm 7 -l {MODEL_NAME}"
# PSM 10 for single character/num recognition
RANK_CONFIG   = f"--psm 10 -l {MODEL_NAME} -c tessedit_char_whitelist=0123456789lLiIoOsSzZ|"
BLACK_TH = 0.01

def process_rank(s: currentState) -> int|None:
    if s.attackLines is None or s.rankCol is None:
        print(f"Error: attackLines or rankCol is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    attack_crop = s.attackLines[s.lineTop:s.lineBottom, s.rankCol.begin:s.rankCol.end]
    rank_preproc = preprocess_line(attack_crop, line=True)

    rank_preproc = rank_preproc[PX_MARGIN : -PX_MARGIN, PX_MARGIN:- PX_MARGIN]
    
    # Specify single character page segmentation, and restrict output to integers only
    rankTXT = pytesseract.image_to_string(rank_preproc, config=RANK_CONFIG)
    rank = auto_correct_num(rankTXT)
    if rank is None:
        print(f"Warning: Could not read rank from image {s.fileNum}. Text: {rankTXT}. Exiting.", file=sys.stderr)
        debug_image(s, rank_preproc, "rank_preproc_error")

    return(rank)

def process_player(s: currentState) -> str:
    if s.attackLines is None or s.playerCol is None:
        print(f"Error: attackLines or playerCol is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    # Crop the player name from the attack lines
    player_preproc = preprocess_line(s.attackLines[s.lineTop:s.lineBottom, s.playerCol.begin:s.playerCol.end], line=True)
    playerTXT = pytesseract.image_to_string(player_preproc, config=PLAYER_CONFIG)
    player = auto_correct_player(s, playerTXT)
    if player is None:
        print(f"Error: Could not read player name from image {s.fileNum}. Text: {playerTXT}. Continuing.", file=sys.stderr)
        debug_image(s, player_preproc, "player_preproc_error")
        debug_image(s, s.attackLines[s.lineTop:s.lineBottom, s.playerCol.begin:s.playerCol.end], "player_crop_error")
        sys.exit(1)
    return player

def process_attack(s: currentState, attackNum: int) -> attackData:
    """Process a single attack line and return an attackData object."""
    if s.attackLines is None or s.enemyCol is None or s.starsCol is None:
        print(f"Error: attackLines, enemyCol or starsCol is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    # Split top half or bottom half of the row depending on attack number
    row_slice   = s.attackLines[s.lineTop:s.lineBottom, :]
    enemy_slice = row_slice[:, s.enemyCol.begin:s.enemyCol.end]
    stars_slice = row_slice[:, s.starsCol.begin:s.starsCol.end]

    attackLine  = np.array_split(enemy_slice,  2, axis=0)
    scoreLine  = np.array_split(cv2.cvtColor(stars_slice, cv2.COLOR_BGR2HLS)[:, :, 1], 2, axis=0)

    # Convert to Lightness and sample minimum to split player rank from name
    attackCrop = cv2.cvtColor(attackLine[attackNum - 1], cv2.COLOR_BGR2HLS)[:, :, 1]
    text_menu_th = sample_image(attackCrop, "max, absolute, minimum, by col", None, eps=0.01) * 0.99
    enemyRankBegin, enemyNameBegin = measure_image(attackCrop, text_menu_th, behavior="absolute threshold, minimum, by col, first fall, next, rise")
    if enemyRankBegin == 0 or enemyNameBegin == 0:
        print(f"Error: Could not detect enemy rank or name begin at positions {enemyRankBegin}, {enemyNameBegin} for absolute threshold minimum of {text_menu_th}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:
            debug_oscilloscope(s, attackCrop.copy(), f"{s.debug_name[0]}_{s.lineNum + s.fileNum}_attack{attackNum}_separating_rank_and_name", None, axis="col")
        sys.exit(1)
    
    # Preprocess original image to read cropped sections using different configurations
    attackPreproc = preprocess_line(attackLine[attackNum - 1], line=True)
    enemyRankTxt  = pytesseract.image_to_string(attackPreproc[:, enemyRankBegin:enemyNameBegin], config=RANK_CONFIG)
    enemy_rank = auto_correct_num(enemyRankTxt)
    if enemy_rank is None:
        print(f"Warning: Could not read enemy rank from image {s.fileNum}. Text: {enemyRankTxt}. Continuing.", file=sys.stderr)
        debug_image(s, attackPreproc[:, enemyRankBegin:enemyNameBegin], "attack_rank_crop_error")

    enemyNameTxt = pytesseract.image_to_string(attackPreproc[:, enemyNameBegin:], config=PLAYER_CONFIG)
    enemy = auto_correct_player(s, enemyNameTxt, enemy=True)
    if enemy is None:
        return(attackData(None, "No attack", "___"))
        # If we've seen this enemy before, use the stored rank
    else:
        if enemy_rank is None and enemy is not None:
            if enemy in s.enemiesSeen:
                enemy_rank = s.enemiesRanks.get(enemy, None)
                # If we haven't seen this enemy before, assume greatest unseen rank
            else:
                ranks = set(s.enemiesRanks.values())
                top = max(ranks) if ranks else 0
                enemy_rank = next((n for n in range(top, 0, -1) if n not in ranks), top + 1)
            print(f"Estimating enemy rank for {enemy.strip('\n')} as {enemy_rank}")

        # Scan vertically to remove white space above and below stars
        starsTop, starsBottom = measure_image(scoreLine[attackNum - 1], BLACK_TH, behavior="stat comparison, min < average, by row, divergence, last, convergence")
        splitStarsLine = scoreLine[attackNum - 1]
        if starsTop == 0 or starsBottom == 0: 
            print(f"Warning: Could not detect top or bottom of stars line in image {s.fileNum}. Missed fixed margin: {BLACK_TH}. Exiting.", file=sys.stderr)
            debug_image(s, splitStarsLine[starsTop:starsBottom, :], f"attack{attackNum}StarsFinalCrop")
            debug_oscilloscope(s, scoreLine[attackNum - 1], f"{s.debug_name}_{str(s.lineNum + s.fileNum)}_stars{attackNum}_y_axis", None, axis="row")

        # Remove a margin of 5 pixels from the top and bottom of the stars line
        if starsTop - STAR_MARGIN > 0: starsTop -= STAR_MARGIN
        if starsBottom + STAR_MARGIN < splitStarsLine.shape[0]: starsBottom += STAR_MARGIN
        # Split the stars line into three parts, each part is a star
        # Each part is 1/3 of the width of the stars line, with a margin of 5 pixels on each side
        stars = np.array_split(splitStarsLine[starsTop:starsBottom, :], 3, axis=1)
        # Score is a 3 character string of stars earned in attack
        score = f"{score_from_stars(stars[0])}{score_from_stars(stars[1])}{score_from_stars(stars[2])}"
        if score.find("☆") != -1 and score.find("★") != -1 and score.find("★") > score.find("☆") or \
           score.find("★") != -1 and score.find("_") != -1 and score.find("★") > score.find("_") or \
           score.find("☆") != -1 and score.find("_") != -1 and score.find("☆") > score.find("_"):
            print(f"Error: Invalid Score of {score}. For image {s.fileNum}, player {s.lineNum}", file=sys.stderr)
            if s.debug_name is not None:
                debug_oscilloscope(s, splitStarsLine[starsTop:starsBottom, :], f"{s.debug_name[0]}_{str(s.lineNum + s.fileNum)}_stars{attackNum}_x_axis", None, axis="col")
            sys.exit(1)

        return(attackData(enemy_rank, enemy, score))


def line_to_player(s: currentState) -> playerData:

    rank = process_rank(s)
    player = process_player(s)

    attack1 = process_attack(s, attackNum=1)
    attack2 = process_attack(s, attackNum=2)

    return playerData(rank, player, [attack1, attack2])

def process_player_data(s: currentState, curr_player: playerData) -> None:
    # If multiaccount detected with identical name, append number to name
    base_name = curr_player.name
    if s.multiAccounters is None:
        print(f"Error: multiAccounters is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    aliases = s.multiAccounters.get(base_name, [])
    # If new player, store attacks and remember in war player array
    if curr_player.name and curr_player.name not in s.playersSeen:
        need_free = (curr_player.rank is None or curr_player.rank >= len(s.war_players) \
                     or s.war_players[curr_player.rank] is not None)
        # If rank misread or already occupied insert at next available placement
        if need_free:                      
            j = 1
            while s.war_players[j] is not None:
                j += 1
            curr_player.rank = j
            print(f"Estimating rank for {curr_player.name.strip('\n')} as {curr_player.rank}.")

        if (aliases and curr_player.rank is not None and s.war_players[curr_player.rank] is None \
            and curr_player.name not in s.playersSeen):
            curr_player.name = aliases.pop(0)
            if not aliases:
                s.multiAccounters[base_name] = []
            if len(s.multiAccounters[base_name]) == 0:
                s.playersSeen.add(base_name)

        if curr_player.rank is not None:
            s.war_players[curr_player.rank] = curr_player
        else:
            print(f"Error: curr_player.rank is None for player {curr_player.name}. Skipping assignment.", file=sys.stderr)
        s.playersSeen.add(curr_player.name)
        if curr_player.attacks is not None:
            for attack in curr_player.attacks:
                if attack.target is not None and attack.target not in s.enemiesSeen:
                    s.enemiesSeen.add(attack.target)
                    s.enemiesRanks[attack.target] = attack.rank
        if s.verbose: print(curr_player.tabulate_player())


def image_to_player_data(s: currentState) -> None:
    """Process the attack lines image to extract player data."""
    if s.attackLines is None or s.attackLinesL is None:
        print(f"Error: attackLines or attackLinesL is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    # Adaptive thresholding for white between lines
    s.linesHeight = s.attackLines.shape[0]
    new_line_TH = sample_image(s.attackLinesL, "max, absolute, minimum, by row", None, eps=0.01) * 0.97

    while True:
        # ---------------------------------------------------- Increment Line ----------------------------------------------------
            s.abs_pos += s.nextLineTop

            s.lineTop = s.abs_pos
            s.lineBottom, s.nextLineTop = measure_image(s.attackLinesL[s.lineTop + PX_MARGIN:, :], new_line_TH, behavior="absolute threshold, minimum, by row, first rise, next, fall")
            if s.nextLineTop == 0:
                print(f"Error: Could not detect bottom of current line or top of next line in image {s.fileNum}. Missing fixed margin: {new_line_TH}. Exiting.", file=sys.stderr); sys.exit(1)
                debug_oscilloscope(s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_top_bottom_margin_error_{s.debug_name[1]}", None, s.OUT_DIR, axis="row")

            if s.nextLineTop == 0:
                s.lineBottom = s.linesHeight
            else:
                s.lineBottom += s.lineTop + PX_MARGIN
                s.nextLineTop += PX_MARGIN  # convert relative → absolute

            s.lineHeight = s.lineBottom - s.lineTop

            curr_player = line_to_player(s)
            process_player_data(s, curr_player)

            if s.lineBottom + s.lineHeight >= s.linesHeight:
                break
            s.lineNum += 1