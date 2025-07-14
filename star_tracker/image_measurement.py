# star_tracker/image_measurement.py
import cv2, numpy as np, sys
from typing import Tuple

from star_tracker.state import currentState, print_to_gui
from star_tracker.preprocessing import measure_image, debug_oscilloscope, sample_image, count_peaks, debug_image
from star_tracker.presets import dataColumn

PX_MARGIN         = 10
OUTLIER_MARGIN    = 15
LOOK_AHEAD_MARGIN = 100

# Scan image by row and column to find menu margins from war background (based on lightness)
def menu_crop(s: currentState) -> np.ndarray:
    """Crop the menu from the background image and return it as a new image: menu."""
    if s.src is None:
        raise ValueError("s.src is None. Cannot convert color.")
    s.srcL = cv2.cvtColor(np.asarray(s.src), cv2.COLOR_BGR2HLS)[:, :, 1]

    if s.measurementPresets is not None:
        m = s.measurementPresets
    if s.presets is not None:
        p = s.presets

    srcH, srcW = s.srcL.shape[:2]
    s.srcDimensions = (srcH, srcW)
    # ------------------------------------------------------------------- Crop menu from background -------------------------------------------------------------------

    # Adaptive thresholding counts the unique jumps in d/dx (avg) which demarcate the menu margins
    menu_col_avg_TH = sample_image(s.srcL, "max, relative, average, by col", 
                                   None, p.col_src_avg_TH.repCharTol) * p.col_src_avg_TH.filterScale
    menu_row_avg_TH = sample_image(s.srcL, "max, relative, average, by row", 
                                   None, p.row_src_avg_TH.repCharTol) * p.row_src_avg_TH.filterScale

    # ------------------------------------------------------------------- Crop Top and bottom margins -------------------------------------------------------------------

    # Scan from top to bottom, by row, to find the jumps in average lightness above the menu background
    menuTopMargin, menuBottomMargin = measure_image(s.srcL, menu_col_avg_TH, 
                                                    behavior="relative threshold, average, by row, first rise, last, fall")
    # If measurement file was created, check if measurements are within expected range
    if s.MEASUREMENT_FILE.exists() and s.debug_name is not None:
        failedTopMargin = m.outside_range(s, menuTopMargin/srcH, "menuTopMargin") or menuTopMargin == 0
        failedBottomMargin = m.outside_range(s, (srcH - menuBottomMargin)/srcH, "menuBottomMargin") or menuBottomMargin >= srcH - 1
        if (failedTopMargin or failedBottomMargin):
            if failedTopMargin and m.menuTopMargin is not None:
                menuTopMargin = m.menuTopMargin.cut
                print_to_gui(s, f"Error: Could not detect Top menu margin in image, Trying previous crop at {menuTopMargin}.")
                
            if failedBottomMargin and m.menuBottomMargin is not None:
                menuBottomMargin = m.menuBottomMargin.cut
                print_to_gui(s, f"Error: Could not detect Bottom menu margin in image, Trying previous crop at {menuBottomMargin}.")

            debug_oscilloscope(s, s.srcL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                            {s.fileNum}_top_bottom_margin_error_{s.debug_name[1]}", [menuTopMargin, menuBottomMargin], axis="row")

    # ------------------------------------------------------------------- Crop left and right margins -------------------------------------------------------------------

    # Scan from left to right, by column, to find the jumps in average lightness above the menu background
    menuLeftMargin, menuRightMargin = measure_image(s.srcL, menu_row_avg_TH, 
                                                    behavior="relative threshold, average, by col, first rise, last, fall")

    # If measurement file was created, check if measurements are within expected range
    if s.MEASUREMENT_FILE.exists() and s.debug_name is not None:
        failedLeftMargin = m.outside_range(s, menuLeftMargin/srcW, "menuLeftMargin") or menuLeftMargin == 0
        failedRightMargin = m.outside_range(s, (srcW - menuRightMargin)/srcW, "menuRightMargin") or menuRightMargin >= srcW - 1
        if (failedLeftMargin or failedRightMargin):
            if failedLeftMargin and m.menuLeftMargin is not None:
                menuLeftMargin = m.menuLeftMargin.cut
                print_to_gui(s, f"Error: Could not detect menu left margin in image, Trying previous crop at {menuLeftMargin}.")

            if failedRightMargin and m.menuRightMargin is not None:
                menuRightMargin = m.menuRightMargin.cut
                print_to_gui(s, f"Error: Could not detect menu right margin in image, Trying previous crop at {menuRightMargin}.")

            debug_oscilloscope(s, s.srcL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                            {s.fileNum}_left_right_margin_error_{s.debug_name[1]}", [menuLeftMargin, menuRightMargin], axis="col")

    # Crop the menu from the background image
    s.menu = s.src[menuTopMargin : menuBottomMargin, menuLeftMargin : menuRightMargin]
    menuH, menuW = s.menu.shape[:2]
    s.menuDimensions = (menuH, menuW)
    s.menuTopMargin = menuTopMargin
    s.menuBottomMargin = menuBottomMargin
    s.menuLeftMargin = menuLeftMargin
    s.menuRightMargin = menuRightMargin
    # ---------------------------------------------------------- Crop attack lines from border of Menu UI ----------------------------------------------------------

    s.menuL = cv2.cvtColor(s.menu, cv2.COLOR_BGR2HLS)[:, :, 1]
    # adaptive thresholding
    col_menu_max_avg_TH = sample_image(s.menuL, "max, absolute, average, by col",
                                       None, s.presets.col_menu_max_avg_TH.repCharTol)* s.presets.col_menu_max_avg_TH.filterScale
    row_menu_min_TH = sample_image(s.menuL, "max, absolute, minimum, by row",
                                   None, s.presets.row_menu_min_TH.repCharTol) * s.presets.row_menu_min_TH.filterScale

    # ---------------------------------------------------- Crop header from menu, keeping space after last line ----------------------------------------------------

    # Scan from top, past the headers to get to the top of the first line, leave the whitespace following the last line
    headerEnd = measure_image(s.menuL[PX_MARGIN:,:], row_menu_min_TH, 
                              behavior="absolute threshold, minimum, by row, first fall, next, fall")[1]
    if s.MEASUREMENT_FILE.exists() and s.debug_name is not None:
        failedHeaderEnd = m.outside_range(s, headerEnd/menuH, "headerEnd") or headerEnd >= menuH - 1
        if failedHeaderEnd:
            if failedHeaderEnd and m.headerEnd is not None:
                headerEnd = m.headerEnd.cut
                print_to_gui(s, f"Error: Could not detect menu left margin in image {s.fileNum}.\
                    Missing fixed margin: {menu_row_avg_TH:.2f}. Trying previous crop at {menuLeftMargin}.") 
                debug_oscilloscope(s, s.menuL.copy(), f"{s.debug_name[0].replace(" ", "_")}_\
                                {s.fileNum}_header_error_{s.debug_name[1]}", [headerEnd], axis="row")

    # Scan from edge of menu to lines, by targetting when average drops below max average
    lineBegin, lineEnd = measure_image(s.menuL[headerEnd:, :], col_menu_max_avg_TH,
                                       behavior=f"absolute threshold, average, by col, first fall, last, rise")
    if s.MEASUREMENT_FILE.exists() and s.debug_name is not None:
        failedLineBegin = m.outside_range(s, lineBegin/menuW, "lineBegin") or lineBegin == 0
        failedLineEnd = m.outside_range(s, (menuW - lineEnd)/menuW, "lineEnd") or lineEnd >= srcW - 1
        if (failedLineBegin or failedLineEnd):
            if failedLineBegin and m.lineBegin is not None:
                lineBegin = m.lineBegin.cut
                print_to_gui(s, f"Error: Could not detect line begin in image, Trying previous crop at {lineBegin}.")

            if failedLineEnd and m.lineEnd is not None:
                lineEnd = m.lineEnd.cut
                print_to_gui(s, f"Error: Could not detect line end in image, Trying previous crop at {lineEnd}.")

            debug_oscilloscope(s, s.menuL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                            {s.fileNum}_line_begin_end_error_{s.debug_name[1]}", [lineBegin, lineEnd], axis="col")

    s.headerEnd = headerEnd
    s.lineBegin = lineBegin
    s.lineEnd = lineEnd

    attackLines = s.menu[headerEnd:, lineBegin:lineEnd]
    alHeight, alWidth = attackLines.shape[:2]
    s.attackLinesDimensions = (alHeight, alWidth)
    # Package the menu as a single lightness channel with the correct dimensions of menu
    return attackLines

def measure_rank(s: currentState, threshold: float) -> None:
    """Measure the rank column in the attack lines image."""
    if s.attackLinesL is None:
        print_to_gui(s, f"Error: attackLines is None for image {s.fileNum}. Exiting.")
        sys.exit(1)

    # Measure the end of the rank column by scanning for the first fall in average lightness
    rankEnd  = measure_image(s.attackLinesL, threshold, 
                             behavior="relative threshold, average, by col, first fall, next, rise")[1]
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None:
        m = s.measurementPresets
        failedRankEnd = m.outside_range(s, rankEnd/s.attackLinesDimensions[1], "rankEnd") or rankEnd >= s.attackLinesDimensions[1] - 1
        if s.debug_name is not None and failedRankEnd:
            if failedRankEnd and m.rankEnd is not None:
                rankEnd = m.rankEnd.cut
                print_to_gui(s, f"Error: Could not detect rank column in image, Trying previous crop at {rankEnd}.")
                debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                                {s.fileNum}_rank_error_{s.debug_name[1]}", [rankEnd], axis="col")
    s.rankEnd = rankEnd
    rankCol = dataColumn(rankEnd)
    s.rankCol = rankCol

def measure_level(s: currentState, threshold: float) -> None:
    """Measure the level column in the attack lines image."""
    if s.attackLinesL is None or s.rankCol is None:
        print_to_gui(s, f"Error: attackLinesL or rankCol is None for image {s.fileNum}. Exiting.")
        sys.exit(1)

    # Level column ends at the first fall in average lightness after the rank column
    levelEnd = measure_image(s.attackLinesL[:, s.rankCol.end:], threshold,
                             behavior="absolute threshold, minimum, by col, first fall, next, fall")[1]
    
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None:
        m = s.measurementPresets
        failedLevelEnd = m.outside_range(s, levelEnd/s.attackLinesDimensions[1], "levelEnd") or levelEnd >= s.attackLinesDimensions[1] - 1
        if s.debug_name is not None and failedLevelEnd:
            if failedLevelEnd and m.levelEnd is not None:
                levelEnd = m.levelEnd.cut
                print_to_gui(s, f"Error: Could not detect level column in image, Trying previous crop at {levelEnd}.")
                debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                                {s.fileNum}_level_error_{s.debug_name[1]}", [levelEnd + s.rankCol.end], axis="col")
    s.levelEnd = levelEnd
    levelCol = dataColumn(levelEnd)
    s.levelCol = levelCol

def measure_player(s:currentState, threshold: float) -> None:
    """Measure the player column in the attack lines image."""
    if s.attackLinesL is None or s.levelCol is None:
        print_to_gui(s, f"Error: attackLinesL or levelCol is None for image {s.fileNum}. Exiting.")
        sys.exit(1)

    # Player column ends at the first fall in average lightness after the level column
    playerEnd = measure_image(s.attackLinesL[:, s.levelCol.end + LOOK_AHEAD_MARGIN:], threshold,
                              behavior="relative threshold, average, by col, from start, next, fall")[1]
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None:
        m = s.measurementPresets
        failedPlayerEnd = m.outside_range(s, (playerEnd + LOOK_AHEAD_MARGIN)/s.attackLinesDimensions[1], "playerEnd") or playerEnd >= s.attackLinesDimensions[1] - 1
        if s.debug_name is not None and failedPlayerEnd:
            if failedPlayerEnd and m.playerEnd is not None:
                playerEnd = m.playerEnd.cut
                print_to_gui(s, f"Error: Could not detect player column in image, Trying previous crop at {playerEnd}.")
                debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                                {s.fileNum}_player_error_{s.debug_name[1]}", [playerEnd + s.levelCol.end], axis="col")
    s.playerEnd = playerEnd     
    playerCol = dataColumn(playerEnd + LOOK_AHEAD_MARGIN)
    s.playerCol = playerCol

def measure_enemy(s: currentState, threshold: float, col_al_global_min_TH: float) -> Tuple[float, int]:
    """Measure the enemy column in the attack lines image."""
    if s.attackLinesL is None or s.playerCol is None:
        print_to_gui(s, f"Error: attackLinesL or playerCol is None for image {s.fileNum}. Exiting.")
        sys.exit(1)
    
    # Scan from the end of the player column to the first presence of black, indicating the start of the enemy column
    enemyStart = measure_image(s.attackLinesL[:, s.playerCol.end:], s.presets.BLACK_TH, 
                               behavior="absolute threshold, minimum, by col, from start, next, rise")[1]
    
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None:
        m = s.measurementPresets
        failedEnemyStart = m.outside_range(s, enemyStart/s.attackLinesDimensions[1], "enemyStart") or enemyStart >= s.attackLinesDimensions[1] - 1
        if s.debug_name is not None and failedEnemyStart:
            if failedEnemyStart and m.enemyStart is not None:
                enemyStart = m.enemyStart.cut
                print_to_gui(s, f"Error: Could not detect enemy column in image, Trying previous crop at {enemyStart}.")
                debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                                {s.fileNum}_enemy_start_error_{s.debug_name[1]}", [enemyStart + s.playerCol.end], axis="col")
    s.enemyStart = enemyStart
    # Look ahead to the final jump in average lightness to find the end of the stars column
    # specifying an additional condition for greater accuracy
    starsColEnd = measure_image(s.attackLinesL[:, s.playerCol.end + PX_MARGIN:], threshold,
                                behavior=f"relative threshold, average, by col, from start, next, rise while min > {col_al_global_min_TH*0.95}")[1]
    
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None:
        m = s.measurementPresets
        failedStarsColEnd = m.outside_range(s, starsColEnd/s.attackLinesDimensions[1], "starsColEnd") or starsColEnd >= s.attackLinesDimensions[1] - 1
        if s.debug_name is not None and failedStarsColEnd:
            if failedStarsColEnd and m.starsColEnd is not None:
                starsColEnd = m.starsColEnd.cut
                print_to_gui(s, f"Error: Could not detect stars column in image, Trying previous crop at {starsColEnd}.")
                debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                                {s.fileNum}_stars_col_end_error_{s.debug_name[1]}", [starsColEnd + s.playerCol.end + PX_MARGIN], axis="col")
    s.starsColEnd = starsColEnd
    starsColEnd = starsColEnd + PX_MARGIN + dataColumn.abs_pos

    # Sample local minimum by filtering out the global max minimum
    col_al_local_min_TH = sample_image(s.attackLinesL[:, enemyStart + PX_MARGIN:starsColEnd - PX_MARGIN], 
                                       "max, absolute, minimum, by col", col_al_global_min_TH,
                                       s.presets.col_al_local_min_TH.repCharTol) * s.presets.col_al_local_min_TH.filterScale

    # Enemy ends when minimum lightness returns to local maximum, skip ahead 100 in case longest enemy rank spacing results in false max
    
    enemyEndSliceStart = s.playerCol.end + enemyStart + LOOK_AHEAD_MARGIN
    enemyEnd_local = measure_image(s.attackLinesL[:, enemyEndSliceStart:],
                             col_al_local_min_TH, behavior=f"absolute threshold, minimum, by col, from start, next, rise")[1]
    enemyEnd_abs = enemyEndSliceStart + enemyEnd_local
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None:
        m = s.measurementPresets
        failedEnemyEnd = m.outside_range(s, (enemyEnd_abs)/s.attackLinesDimensions[1], "enemyEnd") or enemyEnd_abs >= s.attackLinesDimensions[1] - 1
        if s.debug_name is not None and failedEnemyEnd:
            if failedEnemyEnd and m.enemyEnd is not None:
                enemyEnd_abs = m.enemyEnd.cut
                print_to_gui(s, f"Error: Could not detect enemy column in image, Trying previous crop at {enemyEnd_abs}.")
                debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                                {s.fileNum}_enemy_end_error_{s.debug_name[1]}", [enemyEnd_abs], axis="col")
    
    s.enemyEnd = enemyEnd_abs      
    enemyCol = dataColumn(enemyEnd_abs - enemyStart - dataColumn.abs_pos, enemyStart)
    
    s.enemyCol = enemyCol
    
    # Returns local min and starsColEnd for further processing
    return col_al_local_min_TH, starsColEnd

def measure_percentage(s: currentState, threshold: float) -> None:
    """Measure the percentage column in the attack lines image."""
    if s.attackLinesL is None or s.enemyCol is None:
        print_to_gui(s, f"Error: attackLinesL or enemyCol is None for image {s.fileNum}. Exiting.")
        sys.exit(1)
    percentageBegin = measure_image(s.attackLinesL[:, s.enemyCol.end:], threshold,
                                    behavior=f"absolute threshold, minimum, by col, from start, next, fall")[1]
    
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None:
        m = s.measurementPresets
        failedPercentageBegin = m.outside_range(s, (percentageBegin)/s.attackLinesDimensions[1], "percentageBegin") or percentageBegin >= s.attackLinesDimensions[1] - 1
        if s.debug_name is not None and failedPercentageBegin:
            if failedPercentageBegin and m.percentageBegin is not None:
                percentageBegin = m.percentageBegin.cut
                print_to_gui(s, f"Error: Could not detect percentage column in image, Trying previous crop at {percentageBegin}.")
                debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                                {s.fileNum}_percentage_begin_error_{s.debug_name[1]}", [percentageBegin + s.enemyCol.end], axis="col")
    s.percentageBegin = percentageBegin
    # Center the end of enemy column in between the beginning of percentage
    enemyCenter = (percentageBegin//2) + 1
    s.enemyCol.end += enemyCenter
    s.enemyCol.abs_pos += enemyCenter
    percentageBegin -= (percentageBegin//2)
    percentageBegin += s.enemyCol.end

    # First star occurs with presence of white, scan ahead to the first star
    firstStar = measure_image(s.attackLinesL[:, percentageBegin:], s.presets.WHITE_TH,
                              behavior="absolute threshold, maximum, by col, from start, next, rise")[1]
    
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None:
        m = s.measurementPresets
        failedFirstStar = m.outside_range(s, (firstStar)/s.attackLinesDimensions[1], "firstStar") or firstStar >= s.attackLinesDimensions[1] - 1
        if s.debug_name is not None and failedFirstStar:
            if failedFirstStar and m.firstStar is not None:
                firstStar = m.firstStar.cut
                print_to_gui(s, f"Error: Could not detect first star in image, Trying previous crop at {firstStar}.")
                debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                                {s.fileNum}_first_star_error_{s.debug_name[1]}", [firstStar + s.enemyCol.end + percentageBegin], axis="col")
    s.firstStar = firstStar
    # Adjust first star position to be relative to the enemy column
    firstStar += percentageBegin

    # Scan backwards from first star for the first drop in local minimum indicating end of percentage
    starsBegin, percentageEnd = measure_image(cv2.flip(s.attackLinesL[:, percentageBegin:firstStar], 1), threshold,
                                              behavior=f"absolute threshold, minimum, by col, first rise, next, fall")
    
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None and s.debug_name is not None:
        failedstarsBegin = m.outside_range(s, starsBegin/s.attackLinesDimensions[1], "starsBegin") or starsBegin == 0
        failedPercentageEnd = m.outside_range(s, (s.attackLinesDimensions[1] - percentageEnd)/s.attackLinesDimensions[1] - 1, "percentageEnd") or percentageEnd >= s.attackLinesDimensions[1] - 1
        if (failedstarsBegin or failedPercentageEnd):
            if failedstarsBegin and m.starsBegin is not None:
                starsBegin = m.starsBegin.cut
                print_to_gui(s, f"Error: Could not detect stars begin in image, Trying previous crop at {starsBegin}.")

            if failedPercentageEnd and m.percentageEnd is not None:
                percentageEnd = m.percentageEnd.cut
                print_to_gui(s, f"Error: Could not detect percentage end in image, Trying previous crop at {percentageEnd}.")

            debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                            {s.fileNum}_stars_begin_percentage_end_{s.debug_name[1]}", [firstStar - starsBegin, firstStar - percentageEnd], axis="col")
    s.starsBegin = starsBegin
    s.percentageEnd = percentageEnd

    percentageEnd = firstStar - percentageEnd
    starsBegin = firstStar - starsBegin
    # Length returned is the amount to subtract from the end of the percentage column 

    percentageCol = dataColumn(starsBegin - percentageBegin + enemyCenter)
    s.percentageCol = percentageCol

def measure_stars(s: currentState, col_al_local_min_TH: float, starsColEnd: int) -> dataColumn|None:
    """Measure the stars column in the attack lines image."""
    if s.attackLinesL is None or s.percentageCol is None:
        print_to_gui(s, f"Error: attackLinesL or percentageCol is None for image {s.fileNum}. Exiting.")
        sys.exit(1)
    # Scan backwards from explicit attack column end to first presence of black, indicating edge of stars
    realStarsEnd = measure_image(cv2.flip(s.attackLinesL[:, s.percentageCol.end:starsColEnd - PX_MARGIN], 1), 
                                 col_al_local_min_TH ,behavior=f"absolute threshold, minimum, by col, from start, next, fall")[1]
    
    if s.MEASUREMENT_FILE.exists() and s.attackLinesDimensions is not None and s.measurementPresets is not None:
        m = s.measurementPresets
        failedRealStarsEnd = m.outside_range(s, (realStarsEnd)/s.attackLinesDimensions[1], "realStarsEnd") or realStarsEnd >= s.attackLinesDimensions[1] - 1
        if s.debug_name is not None and failedRealStarsEnd:
            if failedRealStarsEnd and m.realStarsEnd is not None:
                realStarsEnd = m.realStarsEnd.cut
                print_to_gui(s, f"Error: Could not detect real stars end in image, Trying previous crop at {realStarsEnd}.")
                debug_oscilloscope(s, s.attackLinesL.copy(), f"{s.debug_name[0].replace(' ', '_')}_\
                                {s.fileNum}_real_stars_end_error_{s.debug_name[1]}", [starsColEnd - PX_MARGIN - realStarsEnd], axis="col")
    s.realStarsEnd = realStarsEnd

    # Adjust realStarsEnd and width to be relative to the percentage column end
    realStarsEnd = starsColEnd - PX_MARGIN - realStarsEnd
    starWidth = realStarsEnd - s.percentageCol.end

    # If no new third star in entire screenshot, realStarsEnd may be the wrong width

    # Two peaks in lightness per star, if less than 6, only a new second star in screenshot
    # if less than 3, then only a new first star in screenshot
    peaks = count_peaks(s.attackLinesL[:, s.percentageCol.end:starsColEnd], s.presets.WHITE_TH)
    # Adjust width to true width of stars based on measured width
    if peaks >= 4 and peaks < 6:
        starWidth = starWidth * (3/2)
    elif peaks < 3:
        starWidth = starWidth * 3

    starsCol = dataColumn(starWidth)
    s.starsCol = starsCol


def measure_data_columns(s: currentState) -> None:
    """Measure the data columns in the attack lines image."""
    if s.attackLines is None:
        raise ValueError("s.attackLines is None. Cannot convert color.")
    s.attackLinesL = cv2.cvtColor(np.asarray(s.attackLines), cv2.COLOR_BGR2HLS)[:, :, 1]

    # Adaptive thresholding counts the unique jumps in d/dx (avg) which demarcate the explicit columns
    # As well as the global minimum, where a jump indicates blank space between data
    col_al_global_min_TH = sample_image(s.attackLinesL[:, OUTLIER_MARGIN:-OUTLIER_MARGIN],
                                        "max, absolute, minimum, by col", None,
                                        s.presets.col_al_global_min_TH.repCharTol)*s.presets.col_al_global_min_TH.filterScale
    
    col_al_sep_TH = sample_image(s.attackLinesL[:, OUTLIER_MARGIN:-OUTLIER_MARGIN],
                                 "max, relative, average, by col", None,
                                 s.presets.col_al_sep_TH.repCharTol)*s.presets.col_al_sep_TH.filterScale

    # Rank begins at 0 and ends at first explicit column
    measure_rank(s, col_al_sep_TH)

    # Level ends at the second explicit column
    measure_level(s, s.presets.BLACK_TH)

    # Player ends at the third explicit column
    measure_player(s, col_al_sep_TH)

    # Enemy begins when first appearance of black and ends when percentage begins at local minimum
    col_al_local_min_TH, starsColEnd = measure_enemy(s, col_al_sep_TH, col_al_global_min_TH)

    # Percentage begins centered between end of enemy and percentage column
    # and ends at the first star, which is the first appearance of white
    measure_percentage(s, col_al_local_min_TH)

    # Stars begin at first appearance of black after percentage column and
    # first appearance of black in reverse from explicit stars column end
    measure_stars(s, col_al_local_min_TH, starsColEnd)

