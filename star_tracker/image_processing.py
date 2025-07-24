# File: star_tracker/image_processing.py
import cv2, numpy as np, pytesseract, sys

from star_tracker.state import currentState, print_to_gui

from star_tracker.player_utils import playerData, attackData
from star_tracker.preprocessing import sample_image, measure_image, debug_image, debug_oscilloscope
from star_tracker.ocr import auto_correct_num, auto_correct_player, score_from_stars, preprocess_line

PX_MARGIN = 10
STAR_MARGIN = 5

def process_rank(s: currentState) -> int|None:
    """Process the rank from the attack lines and return integer rank."""
    if s.attackLines is None or s.rankCol is None:
        print_to_gui(s, f"Error: attackLines or rankCol is None for image \
              {s.fileNum}. Exiting.")
        sys.exit(1)
    # Crop rank from the attack line and preprocess it
    attackCrop = s.attackLines[s.lineTop:s.lineBottom, s.rankCol.begin:s.rankCol.end]
    rankPreproc = preprocess_line(s, attackCrop, line=True)
    
    # Specify single character page segmentation, and restrict output to integers only
    rankTxt = pytesseract.image_to_string(rankPreproc, config=s.RANK_CONFIG)
    rankInt = auto_correct_num(s, rankTxt)

    # If none output, image is likely none as well
    if rankInt is None:
        print_to_gui(s, f"Warning: Could not read rank from image {s.fileNum}. \
              Text: {rankTxt}. Exiting.")
        debug_image(s, rankPreproc, "rank_preproc_error")
    return(rankInt)

def process_player(s: currentState) -> str:
    """Process the player name from the attack lines and return string of player name."""
    if s.attackLines is None or s.playerCol is None:
        print_to_gui(s, f"Error: attackLines or playerCol is None for image \
              {s.fileNum}. Exiting.")
        sys.exit(1)
        
    # Crop the player name from the attack lines and preprocess it
    playerCrop = s.attackLines[s.lineTop:s.lineBottom, s.playerCol.begin:s.playerCol.end]
    playerPreproc = preprocess_line(s, playerCrop, line=True)

    # Specify single line page segmentation, and try to match with submitted player names
    playerTXT = pytesseract.image_to_string(playerPreproc, config=s.PLAYER_CONFIG)
    playerName = auto_correct_player(s, playerTXT, enemy=False, confidence_threshold=s.presets.PLAYERS_CONFIDENCE)
    if playerName is None:
        print_to_gui(s, f"Error: Could not read player name from image {s.fileNum}. \
              Text: {playerTXT}. Continuing.")
        debug_image(s, playerPreproc, "player_preproc_error")
        debug_image(s, playerCrop, "player_crop_error")
        sys.exit(1)
    return playerName

def process_attack(s: currentState, attackNum: int) -> attackData:
    """Process a single attack line and return an attackData object."""
    if s.attackLines is None or s.enemyCol is None or s.starsCol is None:
        print_to_gui(s, f"Error: attackLines, enemyCol or starsCol is None for image \
              {s.fileNum}. Exiting.")
        sys.exit(1)

    # Split top half or bottom half of the row depending on attack number
    rowSlice   = s.attackLines[s.lineTop:s.lineBottom, :]
    enemySlice = rowSlice[:, s.enemyCol.begin:s.enemyCol.end]
    starsSlice = rowSlice[:, s.starsCol.begin:s.starsCol.end]
    print("here")

    # Slice the line in half to separate attacks 1 and 2
    currAttack  = np.array_split(enemySlice,  2, axis=0)[attackNum - 1]
    scoreLine  = np.array_split(cv2.cvtColor(starsSlice, cv2.COLOR_BGR2HLS)[:, :, 1], 2, axis=0)[attackNum - 1]

    # Convert to Lightness and sample minimum from image for threshold
    attackCrop = cv2.cvtColor(currAttack, cv2.COLOR_BGR2HLS)[:, :, 1]
    text_menu_TH = sample_image(attackCrop, "max, absolute, minimum, by col",
                                None, s.presets.text_menu_TH.repCharTol) * s.presets.text_menu_TH.filterScale
    # ------------------------------------------------- Enemy Rank Processing -------------------------------------------------
    # Record the division between enemy rank and name
    enemyRankBegin, enemyNameBegin = measure_image(attackCrop, text_menu_TH, 
                                                   behavior="absolute threshold, minimum, by col, first fall, next, rise")
    if enemyRankBegin == 0 or enemyNameBegin == 0:
        print_to_gui(s, f"Error: Could not detect enemy rank or name begin at positions {enemyRankBegin}, \
              {enemyNameBegin} for absolute threshold minimum of {text_menu_TH}. Exiting.")
        
        if s.debug_name is not None:
            debug_oscilloscope(s, attackCrop.copy(), f"{s.debug_name[0]}_{s.lineNum + s.fileNum}\
                               _attack{attackNum}_separating_rank_and_name", None, axis="col")
        sys.exit(1)
    
    # Preprocess original image to read cropped sections using different page segmentation modes
    attackPreproc = preprocess_line(s, currAttack, line=True)

    # Sample preprocessed image to see if completely white
    preproc_attack_avgL = sample_image(attackPreproc, "avg, absolute, average, by row",
                                       None, s.presets.preproc_attack_avgL.repCharTol) * s.presets.preproc_attack_avgL.filterScale
    # If white, attack line is empty -> no attack
    if preproc_attack_avgL == 1.0:
        print_to_gui(s, f"Warning: No attack data found in image {s.fileNum}. \
              Average Lightness: {preproc_attack_avgL}. Continuing.")
        return(attackData(None, "No attack", "___"))
    else:
        # Pytesseract does its best to read the enemy rank
        enemyRankCrop = attackPreproc[:, enemyRankBegin:enemyNameBegin]
        enemyRankTxt  = pytesseract.image_to_string(enemyRankCrop, config=s.RANK_CONFIG)
        enemy_rank = auto_correct_num(s,enemyRankTxt)
        if enemy_rank is None:
            print_to_gui(s, f"Warning: Could not read enemy rank from image {s.fileNum}. \
                  Text: {enemyRankTxt}. Continuing.")
            debug_image(s, enemyRankCrop, "attack_rank_crop_error")
        # Pytesseract reads the enemy name
        # ------------------------------------------------- Enemy Name Processing -------------------------------------------------
        enemyNameTxt = pytesseract.image_to_string(attackPreproc[:, enemyNameBegin:], config=s.PLAYER_CONFIG)
        enemy = auto_correct_player(s, enemyNameTxt, enemy=True, confidence_threshold=s.presets.ENEMIES_CONFIDENCE)
        if enemy_rank is None and enemy is not None:
            # If we couldn't read the enemy rank, but we have the name, assign it the cannonical rank
            if enemy in s.enemiesSeen:
                enemy_rank = s.enemiesRanks.get(enemy, None)
            # If we haven't seen this enemy before, assume greatest unseen rank
            else:
                ranks = set(s.enemiesRanks.values())
                top = max(ranks) if ranks else 0
                enemy_rank = next((n for n in range(top, 0, -1) if n not in ranks), top + 1)
            print_to_gui(s, f"Estimating enemy rank for {enemy.strip('\n')} as {enemy_rank}")

        # ------------------------------------------------- Enemy Score Processing -------------------------------------------------
        # Scan vertically to remove white space above and below stars
        starsTop, starsBottom = measure_image(scoreLine, s.presets.BLACK_TH, 
                                              behavior="stat comparison, min < average, by row, divergence, last, convergence")
        if starsTop == 0 or starsBottom == 0: 
            print_to_gui(s, f"Warning: Could not detect top or bottom of stars line in image {s.fileNum}. \
                   Missed fixed margin: {s.presets.BLACK_TH}. Exiting.")
            debug_image(s, scoreLine[starsTop:starsBottom, :], f"attack{attackNum}StarsFinalCrop")
            debug_oscilloscope(s, scoreLine, f"{s.debug_name}_{str(s.lineNum + s.fileNum)} \
                               _stars{attackNum}_y_axis", None, axis="row")

        # Remove a margin of 5 pixels from the top and bottom of the stars line
        if starsTop - STAR_MARGIN > 0: starsTop -= STAR_MARGIN
        if starsBottom + STAR_MARGIN < scoreLine.shape[0]: starsBottom += STAR_MARGIN

        # Split the stars line into three parts, each part is a star
        # Each part is 1/3 of the width of the stars line, with a margin of 5 pixels on each side
        stars = np.array_split(scoreLine[starsTop:starsBottom, :], 3, axis=1)

        # Score is a 3 character string of stars earned in attack
        score = f"{score_from_stars(s, stars[0])}{score_from_stars(s, stars[1])}{score_from_stars(s, stars[2])}"
        if score.find("☆") != -1 and score.find("★") != -1 and score.find("★") > score.find("☆") or \
           score.find("★") != -1 and score.find("_") != -1 and score.find("★") > score.find("_") or \
           score.find("☆") != -1 and score.find("_") != -1 and score.find("☆") > score.find("_"):
            print_to_gui(s, f"Error: Invalid Score of {score}. For image {s.fileNum}, player {s.lineNum}")
            if s.debug_name is not None:
                debug_oscilloscope(s, scoreLine[starsTop:starsBottom, :], f"{s.debug_name[0]}_{str(s.lineNum + s.fileNum)}_stars{attackNum}_x_axis", None, axis="col")
            sys.exit(1)

        return(attackData(enemy_rank, enemy, score))


def line_to_player(s: currentState) -> playerData:
    """Process a single line of attack data and return a playerData object."""
    rank = process_rank(s)
    player = process_player(s)

    attack1 = process_attack(s, attackNum=1)
    attack2 = process_attack(s, attackNum=2)

    return playerData(s, rank, player, [attack1, attack2])

def alias_available(canon: str, s: currentState) -> str | None:
    """
    Return the next unused alias for this canon,
    or None if all predefined aliases are taken.
    """
    if s.multiAccounters is None or canon not in s.multiAccounters:
        return None
    variants = s.multiAccounters[canon]          # ["Kit 1", "Kit 2", "Kit 3"]
    used     = s.seenAliases.setdefault(canon, set())
    for v in variants:
        if v not in used:
            return v
    return None 

def process_player_data(s: currentState, currPlayer: playerData) -> None:
    '''Given a playerData Object, file into data structures accordingly.'''
    # If multiaccount detected with identical name, append number to name
    if s.multiAccounters is None:
        print_to_gui(s, f"Error: multiAccounters is None for image {s.fileNum}. Exiting.")
        sys.exit(1)
    canon = None
    if s.aliasMap is not None:
        canon = s.aliasMap.get(currPlayer.name.lower())   # None if not a family we track

    if canon is not None:                       # belongs to a tracked family

        # A) same rank already occupied by this family?  →  reuse stored name
        if (
            currPlayer.rank is not None
            and currPlayer.rank < len(s.war_players)
            and (existing := s.war_players[currPlayer.rank]) is not None
            and s.aliasMap is not None
            and s.aliasMap.get(existing.name.lower()) == canon
        ):
            currPlayer.name = existing.name

        # B) otherwise we need an alias that is still unused
        else:
            alias = alias_available(canon, s)

            if alias is None:
                # We have already used every alias in the JSON (Kit 1-3, James #1-3…)
                # → ignore this extra account entirely.
                return                      # <-- exit process_player_data early
            else:
                currPlayer.name = alias
                s.seenAliases[canon].add(alias)
    # If new player, store attacks and remember in war player array
    if currPlayer.name and currPlayer.name not in s.playersSeen:
        # If player exists, but OCR didnt produce a rank, and rank is currently unseen
        need_free = (currPlayer.rank is None or currPlayer.rank >= len(s.war_players)
                     or s.war_players[currPlayer.rank] is not None)
        # Estimate rank is the next available rank in the war_players array
        if need_free:                      
            j = 1
            while s.war_players[j] is not None:
                j += 1
            currPlayer.rank = j
            print_to_gui(s, f"Estimating rank for {currPlayer.name.strip('\n')} as {currPlayer.rank}.")

        # If a rank was able to be assigned, add player to war_players
        if currPlayer.rank is not None:
            s.war_players[currPlayer.rank] = currPlayer
            s.playersSeen.add(currPlayer.name)
        else:
            print_to_gui(s, f"Error: currPlayer.rank is None for player {currPlayer.name}. \
                  Skipping assignment.")
            sys.exit(1)
        
        # Add the current player's targets to the enemiesSeen set and dictionary
        if currPlayer.attacks is not None:
            for attack in currPlayer.attacks:
                # If new enemy exists but does not have a rank, but was seen before, assign it the global rank
                if attack.rank is None:
                    if attack.target in s.enemiesSeen:
                        attack.rank = s.enemiesRanks.get(attack.target, None)
                    else:
                        # And the rank is None, try 
                        j = 1
                        while j < len(s.war_enemies) and s.war_enemies[j] is not None:
                            j += 1
                        attack.rank = j
                        s.war_enemies[j] = attack.target
                if attack.rank is not None:
                    s.war_enemies[attack.rank] = attack.target
                    s.enemiesRanks[attack.target] = attack.rank
                s.enemiesSeen.add(attack.target)
        print_to_gui(s, currPlayer.tabulate_player())


def image_to_player_data(s: currentState) -> None:
    '''Process the attack lines image to extract player data.'''
    if s.attackLines is None or s.attackLinesL is None:
        print_to_gui(s, f"Error: attackLines or attackLinesL is None for image {s.fileNum}. Exiting.")
        sys.exit(1)
    # Height of total menu lines
    s.linesHeight = s.attackLines.shape[0]
    # Adaptive thresholding for space between lines
    new_line_TH = sample_image(s.attackLinesL, "max, absolute, minimum, by row",
                               None, s.presets.new_line_TH.repCharTol) * s.presets.new_line_TH.filterScale

    while True:
        # Update absolute position of the to the top of the next line
        s.abs_pos += s.nextLineTop
        # Set top of current line to the previously next top
        s.lineTop = s.abs_pos

        # When minimum rises, end of line is reached, when it falls, next line is reached
        s.lineBottom, s.nextLineTop = measure_image(s.attackLinesL[s.lineTop + PX_MARGIN:, :], 
                                                    new_line_TH, behavior="absolute threshold, minimum, by row, first rise, next, fall")
        if s.nextLineTop == 0:
            print_to_gui(s, f"Error: Could not detect bottom of current line or top of next line in image \
                    {s.fileNum}. Missing fixed margin: {new_line_TH}. Exiting."); sys.exit(1)
            debug_oscilloscope(s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_top_bottom_margin_error\
                                _{s.debug_name[1]}", None, s.OUT_DIR, axis="row")

        # White space is kept after the final line, however if next line not found,
        # assume the end of the image and crop to the absolute bottom of the image
        if s.nextLineTop == 0:
            s.lineBottom = s.linesHeight
        # Otherwise, update the line bottom to the next line top
        # and add a margin to the bottom of the line
        else:
            s.lineBottom += s.lineTop + PX_MARGIN
            s.nextLineTop += PX_MARGIN  # convert relative → absolute

        s.lineHeight = s.lineBottom - s.lineTop

        # Iterators are all recorded within current state and passed to processing functions
        currPlayer = line_to_player(s)
        process_player_data(s, currPlayer)

        if s.lineBottom + s.lineHeight >= s.linesHeight:
            break
        s.lineNum += 1