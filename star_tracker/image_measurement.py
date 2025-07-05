import cv2, numpy as np, sys
from .data_structures import dataColumn, currentState
from .preprocessing import dataColumn, measure_image, debug_oscilloscope, sample_image, count_peaks, debug_image

from typing import Tuple

MENU_BG_TH        = 0.3
PX_MARGIN         = 10
OUTLIER_MARGIN    = 15
LOOK_AHEAD_MARGIN = 100

BLACK_TH          = 0.01
WHITE_TH          = 0.99

# Scan image by row and column to find menu margins from war background (based on lightness)
def menu_crop(s: currentState) -> np.ndarray:
    if s.src is None:
        raise ValueError("s.src is None. Cannot convert color.")
    s.srcL = cv2.cvtColor(np.asarray(s.src), cv2.COLOR_BGR2HLS)[:, :, 1]

    srcH, srcW = s.srcL.shape[:2]
    # ------------------------------------------------------------------- Crop menu from background -------------------------------------------------------------------
    menuTopMargin, menuBottomMargin = measure_image(s.srcL, MENU_BG_TH, behavior="relative threshold, average, by row, first rise, last, fall")
    if menuTopMargin == 0 or menuBottomMargin == srcH: 
        print(f"Error: Could not detect menu margins in image {s.fileNum}. Missing fixed margin: {MENU_BG_TH}. Exiting.", file=sys.stderr); sys.exit(1)
        debug_oscilloscope(s.srcL.copy(), f"{s.debug_name[0]}_{s.fileNum}_top_bottom_margin_error_{s.debug_name[1]}", None, s.OUT_DIR, axis="row")

    menuLeftMargin, menuRightMargin = measure_image(s.srcL, MENU_BG_TH, behavior="relative threshold, average, by col, first rise, last, fall")
    if menuLeftMargin == 0 or menuRightMargin == srcW: 
        print(f"Error: Could not detect menu margins in image {s.fileNum}. Missing fixed margin: {MENU_BG_TH}. Exiting.", file=sys.stderr); sys.exit(1)
        debug_oscilloscope(s.srcL.copy(), f"{s.debug_name[0]}_{s.fileNum}_left_right_margin_error_{s.debug_name[1]}", None, s.OUT_DIR, axis="col")

    menu = s.src[menuTopMargin : menuBottomMargin, menuLeftMargin : menuRightMargin]
    menuH, menuW = menu.shape[:2] 

    # ---------------------------------------------------------- Crop Border of menu to just the attack lines ----------------------------------------------------------

    menuL = cv2.cvtColor(menu, cv2.COLOR_BGR2HLS)[:, :, 1]
    # adaptive thresholding
    col_menu_max_avg_TH = sample_image(menuL, "max, absolute, average, by col", None, eps=0.001)*0.99
    row_menu_min_TH = sample_image(menuL, "max, absolute, minimum, by row", None, eps=0.001) * 0.97

    # Scan from top, past the headers to get to the top of the first line, leave the whitespace following the last line
    headerEnd = measure_image(menuL[PX_MARGIN:,:], row_menu_min_TH, behavior="absolute threshold, minimum, by row, first fall, next, fall")[1]
    if headerEnd <= menuH * 0.2 or headerEnd == menuH: 
        print(f"Error: Could not detect header in image {s.fileNum}. Missing absolute threshold minimum: {row_menu_min_TH}. \
              Reporting header of {headerEnd} at {headerEnd/menuH:.2%} of menu. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, menuL.copy(), f"{s.debug_name[0]}_{s.fileNum}_header_error_{s.debug_name[1]}", None, axis="row")
        sys.exit(1)

    # Scan from edge of menu to lines, by targetting when average drops below max average
    lineBegin, lineEnd = measure_image(menuL[headerEnd:, :], col_menu_max_avg_TH, behavior=f"absolute threshold, average, by col, first fall, last, rise")
    if lineBegin == 0 or lineEnd == menuW: 
        print(f"Error: Could not detect attack lines in image {s.fileNum}. Missing absolute threshold average: {col_menu_max_avg_TH}. \
              Reporting lineBegin of {lineBegin} and lineEnd of {lineEnd}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, menuL.copy(), f"{s.debug_name[0]}_{s.fileNum}_line_begin_end_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)

    # Package the menu as a single lightness channel with the correct dimensions of menu
    return menu[headerEnd:, lineBegin:lineEnd]

def measure_rank(s: currentState, threshold: float) -> None:
    if s.attackLinesL is None:
        print(f"Error: attackLines is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    rankEnd  = measure_image(s.attackLinesL, threshold, behavior="relative threshold, average, by col, first fall, next, rise")[1]
    rankCol = dataColumn(rankEnd)

    if rankEnd == 0 or rankEnd > s.attackLinesL.shape[1] * 0.1:
        print(f"Error: Could not detect rank column end at position {rankEnd} for relative threshold average of {threshold}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_rank_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)
    s.rankCol = rankCol

def measure_level(s: currentState, threshold: float) -> None:
    if s.attackLinesL is None or s.rankCol is None:
        print(f"Error: attackLinesL or rankCol is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    levelEnd = measure_image(s.attackLinesL[:, s.rankCol.end:], threshold, behavior="absolute threshold, minimum, by col, first fall, next, fall")[1]
    levelCol = dataColumn(levelEnd)
    
    if levelEnd == 0 or levelEnd > s.attackLinesL.shape[1] * 0.06:
        print(f"Error: Could not detect level column end at position {levelEnd} for absolute threshold minimum of {threshold}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_level_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)
    s.levelCol = levelCol

def measure_player(s:currentState, threshold: float) -> None:
    if s.attackLinesL is None or s.levelCol is None:
        print(f"Error: attackLinesL or levelCol is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    playerEnd = measure_image(s.attackLinesL[:, s.levelCol.end + LOOK_AHEAD_MARGIN:], threshold, behavior="relative threshold, average, by col, from start, next, fall")[1]
    playerCol = dataColumn(playerEnd + LOOK_AHEAD_MARGIN)

    if playerEnd == 0 or playerEnd > s.attackLinesL.shape[1] * 0.3:
        print(f"Error: Could not detect player column end at position {playerEnd} for relative threshold average of {threshold}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_player_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)
    s.playerCol = playerCol

def measure_enemy(s: currentState, threshold: float, global_min_TH: float) -> Tuple[float, int]:
    if s.attackLinesL is None or s.playerCol is None:
        print(f"Error: attackLinesL or playerCol is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    enemyStart = measure_image(s.attackLinesL[:, s.playerCol.end:], BLACK_TH, behavior="absolute threshold, minimum, by col, from start, next, rise")[1]

    if enemyStart == 0 or enemyStart > s.attackLinesL.shape[1] * 0.4:
        print(f"Error: Could not detect enemy column start at position {enemyStart} for absolute threshold minimum of {threshold}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_enemy_start_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)

    starsColEnd = measure_image(s.attackLinesL[:, s.playerCol.end + PX_MARGIN:], threshold, behavior=f"relative threshold, average, by col, from start, next, rise while min > {global_min_TH*0.95}")[1]
    if starsColEnd == 0 or starsColEnd > s.attackLinesL.shape[1] * 0.55:
        print(f"Error: Could not detect stars column end at position {starsColEnd} for relative threshold average of {threshold}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_stars_end_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)
        
    starsColEnd = starsColEnd + PX_MARGIN + dataColumn.abs_pos

    local_max_TH = sample_image(s.attackLinesL[:, enemyStart + PX_MARGIN:starsColEnd - PX_MARGIN], "max, absolute, minimum, by col", global_min_TH, eps=0.01) * 0.95

    # Enemy ends when minimum lightness returns to local maximum, skip ahead 50 in case longest enemy rank spacing results in false max
    enemyEnd = measure_image(s.attackLinesL[:, s.playerCol.end + enemyStart + LOOK_AHEAD_MARGIN:], local_max_TH, behavior=f"absolute threshold, minimum, by col, from start, next, rise")[1]
    enemyCol = dataColumn(enemyEnd + PX_MARGIN + LOOK_AHEAD_MARGIN, enemyStart - PX_MARGIN)
    if enemyEnd == 0 or enemyEnd > s.attackLinesL.shape[1] * 0.4:
        print(f"Error: Could not detect enemy column end at position {enemyEnd} for absolute threshold minimum of {local_max_TH}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_enemy_end_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)
    enemyCol.begin -= PX_MARGIN

    s.enemyCol = enemyCol
    return local_max_TH, starsColEnd

def measure_percentage(s: currentState, threshold: float) -> None:
    if s.attackLinesL is None or s.enemyCol is None:
        print(f"Error: attackLinesL or enemyCol is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    percentageBegin = measure_image(s.attackLinesL[:, s.enemyCol.end:], threshold, behavior=f"absolute threshold, minimum, by col, from start, next, fall")[1]
    # Center the end of enemy column in between the beginning of percentage
    enemyCenter = (percentageBegin//2) + 1
    s.enemyCol.end += enemyCenter
    s.enemyCol.abs_pos += enemyCenter
    percentageBegin -= (percentageBegin//2)
    percentageBegin += s.enemyCol.end

    # First star occurs with presence of white, scan ahead to the first star
    firstStar = measure_image(s.attackLinesL[:, percentageBegin:], WHITE_TH, behavior="absolute threshold, maximum, by col, from start, next, rise")[1]
    if firstStar == 0 or firstStar > s.attackLinesL.shape[1] * 0.06:
        print(f"Error: Could not detect first star at position {firstStar} for absolute threshold maximum of {WHITE_TH}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_first_star_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)
    firstStar += percentageBegin

    # Scan backwards from first star for the first drop in local minimum indicating end of percentage
    starsBegin, percentageEnd = measure_image(cv2.flip(s.attackLinesL[:, percentageBegin:firstStar], 1), threshold ,behavior=f"absolute threshold, minimum, by col, first rise, next, fall")
    percentageEnd = firstStar - percentageEnd
    starsBegin = firstStar - starsBegin
    percentageCol = dataColumn(starsBegin - percentageBegin + enemyCenter)
    if percentageEnd == 0 or starsBegin == 0 or starsBegin - percentageEnd > 0.1 * s.attackLinesL.shape[1]:
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_percentage_stars_error_{s.debug_name[1]}", None, axis="col")
        print(f"Error: Could not detect percentage end or stars begin at positions {percentageEnd}, {starsBegin} for absolute threshold minimum of {threshold}. Exiting.", file=sys.stderr)
        sys.exit(1)
    s.percentageCol = percentageCol

def measure_stars(s: currentState, local_max_TH: float, starsColEnd: int) -> dataColumn|None:
    if s.attackLinesL is None or s.percentageCol is None:
        print(f"Error: attackLinesL or percentageCol is None for image {s.fileNum}. Exiting.", file=sys.stderr)
        sys.exit(1)
    # Scan backwards from explicit attack column end to first presence of black, indicating edge of stars
    realStarsEnd = measure_image(cv2.flip(s.attackLinesL[:, s.percentageCol.end:starsColEnd - PX_MARGIN], 1), local_max_TH ,behavior=f"absolute threshold, minimum, by col, from start, next, fall")[1]
    if realStarsEnd == 0 or realStarsEnd > s.attackLinesL.shape[1] * 0.15:
        print(f"Error: Could not detect real stars end at position {realStarsEnd} for absolute threshold minimum of {local_max_TH}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_real_stars_end_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)

    realStarsEnd = starsColEnd - PX_MARGIN - realStarsEnd
    starWidth = realStarsEnd - s.percentageCol.end

    # If no new third star, realStarsEnd may be the wrong width
    # Two peaks in lightness per star, if less than 6, only a new second star in screenshot
    # if less than 3, then only a new first star in screenshot
    peaks = count_peaks(s.attackLinesL[:, s.percentageCol.end:starsColEnd], WHITE_TH)
    # Adjust width to true width of stars based on measured width
    if peaks >= 4 and peaks < 6:
        starWidth = starWidth * (3/2)
    elif peaks < 3:
        starWidth = starWidth * 3
    
    starsCol = dataColumn(starWidth)
    if starWidth == 0 or starWidth > s.attackLinesL.shape[1] * 0.08:
        print(f"Error: Could not detect stars column width at position {starWidth} for absolute threshold minimum of {local_max_TH}. Exiting.", file=sys.stderr)
        if s.debug_name is not None:    
            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0]}_{s.fileNum}_stars_col_error_{s.debug_name[1]}", None, axis="col")
        sys.exit(1)

    s.starsCol = starsCol


def measure_data_columns(s: currentState) -> None:
    if s.attackLines is None:
        raise ValueError("s.attackLines is None. Cannot convert color.")
    s.attackLinesL = cv2.cvtColor(np.asarray(s.attackLines), cv2.COLOR_BGR2HLS)[:, :, 1]

    # Adaptive thresholding counts the unique jumps in d/dx (avg) which demarcate the explicit columns
    # As well as the global minimum, where a jump indicates blank space between data
    global_min_TH = sample_image(s.attackLinesL[:, OUTLIER_MARGIN:-OUTLIER_MARGIN], "max, absolute, minimum, by col", None, eps=0.001)*0.99
    col_sep_TH = sample_image(s.attackLinesL[:, OUTLIER_MARGIN:-OUTLIER_MARGIN], "max, relative, average, by col", None, eps=0.0005)*0.99

    # Rank begins at 0 and ends at first explicit column
    measure_rank(s, col_sep_TH)

    # Level ends at the second explicit column
    measure_level(s, BLACK_TH)

    # Player ends at the third explicit column
    measure_player(s, col_sep_TH)

    # Enemy begins when first appearance of black in fourth explicit column
    local_max_TH, starsColEnd = measure_enemy(s, col_sep_TH, global_min_TH)

    # Percentage ends when stars begin
    measure_percentage(s, local_max_TH)

    measure_stars(s, local_max_TH, starsColEnd)

